/**
 * predict_native_v2.cpp
 * T-regressor ネイティブ推論器 (外部依存ゼロ)
 *
 * 対応モデル: Linear (Ridge), LightGBM, GP (ARD-RBF), MLP (sklearn),
 *            Linear-Poly (多項式Ridge), Blend (アンサンブル、他モデルの入れ子)
 * モデル形式: .treg バイナリ (EXE テールへの自己埋め込みまたはファイル指定)
 *
 * EXE テール形式:
 *   [EXE bytes] [treg bytes] [uint64_le treg_size] [8 bytes "TREG_EMB"]
 *
 * ビルド (MSVC):
 *   cl /O2 /EHsc /std:c++17 predict_native_v2.cpp /Fe:predict_native.exe
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <iomanip>
#include <memory>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
static void fatal(const char* msg) {
    int wlen = MultiByteToWideChar(CP_UTF8, 0, msg, -1, NULL, 0);
    if (wlen > 0) {
        std::wstring w(wlen, 0);
        MultiByteToWideChar(CP_UTF8, 0, msg, -1, &w[0], wlen);
        MessageBoxW(NULL, w.c_str(), L"T-regressor", MB_OK | MB_ICONERROR);
    }
}
#else
static void fatal(const char* msg) { fprintf(stderr, "ERROR: %s\n", msg); }
#endif

// ── マジック ──────────────────────────────────────────────────────────────────
static const char TREG_MAGIC[4]  = {'T','R','E','G'};
static const char EXE_MAGIC[8]   = {'T','R','E','G','_','E','M','B'};
enum ModelType  { MT_LINEAR = 0, MT_LGBM = 1, MT_GP = 2, MT_MLP = 3,
                  MT_LINEAR_POLY = 4, MT_BLEND = 5 };
enum YTransform { YT_NONE = 0, YT_LOG1P = 1, YT_YEO_JOHNSON = 2 };

// ── バイナリリーダー（境界チェック付き） ──────────────────────────────────────
struct Reader {
    const uint8_t* data;
    size_t pos, size;
    bool fail = false;

    bool ok()  const { return !fail && pos <= size; }
    bool eof() const { return pos >= size; }

    template<typename T>
    T read() {
        if (fail || pos + sizeof(T) > size) { fail = true; return T{}; }
        T v; std::memcpy(&v, data + pos, sizeof(T)); pos += sizeof(T); return v;
    }

    std::string read_str() {
        uint16_t len = read<uint16_t>();
        if (fail || pos + len > size) { fail = true; return {}; }
        std::string s(reinterpret_cast<const char*>(data + pos), len);
        pos += len;
        return s;
    }

    std::vector<float> read_floats(size_t n) {
        if (fail || n > (size_t)1e8 || pos + n * sizeof(float) > size) {
            fail = true; return {};
        }
        std::vector<float> v(n);
        std::memcpy(v.data(), data + pos, n * sizeof(float));
        pos += n * sizeof(float);
        return v;
    }
};

// ── モデル構造体 ──────────────────────────────────────────────────────────────
struct LinearModel {
    std::vector<float> mean, scale, coef;
    float intercept = 0;
};

struct GPModel {
    std::vector<float> mean, scale, ls;
    float sv = 1, y_mean = 0, y_std = 1;
    int n_train = 0, n_feat = 0;
    std::vector<float> X_train, alpha;
};

struct MLPLayer {
    int n_in = 0, n_out = 0;
    uint8_t act = 0;             // 0=relu, 1=identity
    std::vector<float> W, b;    // W: [n_in * n_out] row-major (sklearn 順)
};

struct MLPModel {
    std::vector<float> mean, scale;
    std::vector<MLPLayer> layers;
};

struct LGBMTree {
    int n_leaves = 0;
    std::vector<uint32_t> split_feature;
    std::vector<float>    threshold;
    std::vector<int32_t>  left_child, right_child;
    std::vector<float>    leaf_value;
};

struct LGBMModel { std::vector<LGBMTree> trees; };

// linear_poly (多項式Ridge): RobustScaler相当のcenter/scale → 多項式項(単項 or
// 標準化後の値どうしの積/二乗) → coef内積 + intercept。項の並びは
// _light.PolynomialFeatures.transform と同一(train_bridge._write_treg_stream参照)、
// term_b<0 は「単項(term_aそのまま)」を表す。
struct LinearPolyModel {
    std::vector<float>   center, scale;
    std::vector<int32_t> term_a, term_b;
    std::vector<float>   coef;
    float intercept = 0;
};

// blend (アンサンブル): 各メンバーは後処理なし(smear=1, y_clip=無制限, round無し)の
// 自己完結した入れ子 .treg ブロブとして埋め込まれている(train_bridge._export_treg_blend
// 参照)。TregModel はまだ完全定義されていないため、メンバーは unique_ptr で保持する
// (再帰的な木構造。JS版 predict-core.js の loadTreg/predictRow と同じ設計)。
struct TregModel;
struct BlendMember {
    float weight = 1.0f;
    std::unique_ptr<TregModel> model;
};
struct BlendModel { std::vector<BlendMember> members; };

// v4: 自動特徴量（学習時に生成した派生特徴のレシピ）
enum DerivedOp { DOP_MUL = 0, DOP_SQ = 1, DOP_SIGN = 2 };
struct DerivedFeat {
    uint8_t op = 0;
    std::string name, col_a, col_b;
    float a_lo = -3.4e38f, a_hi = 3.4e38f;
    float b_lo = -3.4e38f, b_hi = 3.4e38f;
};

struct TregModel {
    uint8_t    file_version = 1;
    ModelType  type         = MT_LGBM;
    YTransform y_transform  = YT_NONE;
    float      yeo_lambda   = 1.0f;
    int n_feat = 0;
    std::string target_col;
    std::vector<std::string> feat_cols;
    std::map<std::string, double> medians;
    std::vector<DerivedFeat> derived;
    std::map<std::string, size_t> derived_idx;   // out_name → derived index

    // v3: 予測後処理（整数丸め / smearing補正 / Y観測レンジclip / Xクリップ）
    uint8_t round_output = 0;
    float   smear        = 1.0f;
    float   y_clip_lo    = -3.4e38f, y_clip_hi = 3.4e38f;
    std::vector<float> x_clip_lo, x_clip_hi;

    LinearModel     linear;
    GPModel         gp;
    MLPModel        mlp;
    LGBMModel       lgbm;
    LinearPolyModel linear_poly;
    BlendModel      blend;
};

// ── .treg デシリアライズ ──────────────────────────────────────────────────────
static bool load_treg(const uint8_t* data, size_t size, TregModel& out) {
    Reader r{data, 0, size};
    if (size < 6) return false;
    char magic[4]; std::memcpy(magic, r.data, 4); r.pos += 4;
    if (std::memcmp(magic, TREG_MAGIC, 4) != 0) return false;

    out.file_version = r.read<uint8_t>();
    out.type  = (ModelType)r.read<uint8_t>();
    out.n_feat = (int)r.read<uint32_t>();
    const int d = out.n_feat;

    // サニティチェック: 破損・未来バージョンのファイルを明示拒否
    if (out.file_version > 4) {
        fatal("このexeより新しいモデル形式です。\n最新版のT-regressorで書き出したexeを使用してください。");
        return false;
    }
    // blend(アンサンブル)は自身の直接の特徴ベクトルを持たない(各メンバーが個別に持つ)ため
    // n_feat=0 が正当な値になる(JS版 predict-core.js loadTreg と同じ判定)。
    if (d > 100000 || (out.type != MT_BLEND && d < 1)) return false;

    // v4: 派生特徴レシピ（自動FE）
    if (out.file_version >= 4) {
        uint32_t n_derived = r.read<uint32_t>();
        if (r.fail || n_derived > 100000) return false;
        out.derived.resize(n_derived);
        for (uint32_t i = 0; i < n_derived; i++) {
            auto& df = out.derived[i];
            df.op    = r.read<uint8_t>();
            df.name  = r.read_str();
            df.col_a = r.read_str();
            df.a_lo  = r.read<float>();
            df.a_hi  = r.read<float>();
            df.col_b = r.read_str();
            df.b_lo  = r.read<float>();
            df.b_hi  = r.read<float>();
            if (r.fail || df.op > 2) return false;
            out.derived_idx[df.name] = i;
        }
    }

    if (out.type == MT_LINEAR) {
        out.linear.mean  = r.read_floats(d);
        out.linear.scale = r.read_floats(d);
        out.linear.coef  = r.read_floats(d);
        out.linear.intercept = r.read<float>();

    } else if (out.type == MT_LGBM) {
        uint32_t n_trees = r.read<uint32_t>();
        /*reserved=*/     r.read<uint32_t>();
        if (r.fail || n_trees > 100000) return false;
        out.lgbm.trees.resize(n_trees);
        for (auto& tree : out.lgbm.trees) {
            tree.n_leaves = (int)r.read<uint32_t>();
            if (r.fail || tree.n_leaves < 1 || tree.n_leaves > (1 << 20)) return false;
            int ni = tree.n_leaves - 1;   // n_leaves==1 (単葉=定数寄与) は内部ノード0個
            tree.split_feature.resize(ni);
            for (auto& x : tree.split_feature) x = r.read<uint32_t>();
            tree.threshold.resize(ni);
            for (auto& x : tree.threshold)    x = r.read<float>();
            tree.left_child.resize(ni);
            for (auto& x : tree.left_child)   x = r.read<int32_t>();
            tree.right_child.resize(ni);
            for (auto& x : tree.right_child)  x = r.read<int32_t>();
            tree.leaf_value.resize(tree.n_leaves);
            for (auto& x : tree.leaf_value)   x = r.read<float>();
            if (r.fail) return false;
        }

    } else if (out.type == MT_GP) {
        out.gp.n_feat = d;
        out.gp.mean   = r.read_floats(d);
        out.gp.scale  = r.read_floats(d);
        out.gp.ls     = r.read_floats(d);
        out.gp.sv     = r.read<float>();
        out.gp.y_mean = r.read<float>();
        out.gp.y_std  = r.read<float>();
        out.gp.n_train = (int)r.read<uint32_t>();
        if (r.fail || out.gp.n_train < 1 ||
            (size_t)out.gp.n_train * (size_t)d > (size_t)1e8) return false;
        out.gp.X_train = r.read_floats((size_t)out.gp.n_train * d);
        out.gp.alpha   = r.read_floats(out.gp.n_train);

    } else if (out.type == MT_MLP) {
        out.mlp.mean  = r.read_floats(d);
        out.mlp.scale = r.read_floats(d);
        uint32_t n_layers = r.read<uint32_t>();
        if (r.fail || n_layers < 1 || n_layers > 64) return false;
        out.mlp.layers.resize(n_layers);
        for (auto& layer : out.mlp.layers) {
            layer.n_in  = (int)r.read<uint32_t>();
            layer.n_out = (int)r.read<uint32_t>();
            layer.act   = r.read<uint8_t>();
            if (r.fail || layer.n_in < 1 || layer.n_out < 1 ||
                (size_t)layer.n_in * (size_t)layer.n_out > (size_t)1e8) return false;
            layer.W = r.read_floats((size_t)layer.n_in * layer.n_out);
            layer.b = r.read_floats(layer.n_out);
        }
    } else if (out.type == MT_LINEAR_POLY) {
        out.linear_poly.center = r.read_floats(d);
        out.linear_poly.scale  = r.read_floats(d);
        uint32_t n_terms = r.read<uint32_t>();
        if (r.fail || n_terms > 1000000) return false;
        out.linear_poly.term_a.resize(n_terms);
        out.linear_poly.term_b.resize(n_terms);
        for (uint32_t i = 0; i < n_terms; i++) {
            int32_t ta = r.read<int32_t>();
            int32_t tb = r.read<int32_t>();
            // term_b<0 は「単項」を表す正当な値。それ以外は両方とも [0, d) の範囲内で
            // なければならない。信頼できるwriter(train_bridge._write_treg_stream)は
            // 常にこの範囲内の値しか書かないが、壊れた.treg/将来のwriterのバグに対して
            // predict_linear_poly() 側の s[a]/s[b] が範囲外読み出しにならないよう
            // ここで明示的に拒否する(他の可変長フィールドと同様の防御)。
            if (r.fail || ta < 0 || ta >= d || (tb >= 0 && tb >= d)) return false;
            out.linear_poly.term_a[i] = ta;
            out.linear_poly.term_b[i] = tb;
        }
        out.linear_poly.coef = r.read_floats(n_terms);
        out.linear_poly.intercept = r.read<float>();

    } else if (out.type == MT_BLEND) {
        // 各メンバーは「重み(f32) + ブロブ長(u32) + 自己完結した入れ子.treg」の並び。
        // load_treg をブロブ範囲に対して再帰呼び出しし、読み終えたら r.pos をブロブ長分
        // 進める(ネストしたReaderはブロブ先頭からの相対posで独立に動くため)。
        uint32_t n_members = r.read<uint32_t>();
        if (r.fail || n_members > 1000) return false;
        for (uint32_t i = 0; i < n_members; i++) {
            float weight = r.read<float>();
            uint32_t blob_len = r.read<uint32_t>();
            if (r.fail || r.pos + blob_len > r.size) return false;
            auto member = std::make_unique<TregModel>();
            if (!load_treg(r.data + r.pos, blob_len, *member)) return false;
            r.pos += blob_len;
            out.blend.members.push_back(BlendMember{weight, std::move(member)});
        }
        if (out.blend.members.size() < 2) return false;

    } else {
        return false;  // 未知のモデル型
    }
    if (r.fail) return false;

    // Y 逆変換情報 (v2+)
    if (out.file_version >= 2) {
        out.y_transform = (YTransform)r.read<uint8_t>();
        if (out.y_transform == YT_YEO_JOHNSON)
            out.yeo_lambda = r.read<float>();
    }

    // 予測後処理情報 (v3+): 整数丸め / smearing補正 / Y観測レンジclip / Xクリップ
    if (out.file_version >= 3) {
        out.round_output = r.read<uint8_t>();
        out.smear        = r.read<float>();
        out.y_clip_lo    = r.read<float>();
        out.y_clip_hi    = r.read<float>();
        uint32_t n_fc_clip = r.read<uint32_t>();
        // 他の可変長配列(n_fc/n_med等)と同様の上限検査が無かったため、破損した.tregで
        // n_fc_clipに巨大な値が入っていると resize() が std::bad_alloc/length_error を
        // 送出し、main()側で未捕捉のまま abort()(exit=134、-mwindowsビルドでは
        // メッセージも出ず無言クラッシュ)していた(低-M12)。
        if (r.fail || n_fc_clip > 100000) return false;
        out.x_clip_lo.resize(n_fc_clip);
        out.x_clip_hi.resize(n_fc_clip);
        for (uint32_t i = 0; i < n_fc_clip; i++) {
            out.x_clip_lo[i] = r.read<float>();
            out.x_clip_hi[i] = r.read<float>();
        }
    }

    // 共通テール: target_col, feat_cols, medians
    out.target_col = r.read_str();
    uint32_t n_fc = r.read<uint32_t>();
    if (r.fail || n_fc > 100000) return false;
    out.feat_cols.resize(n_fc);
    for (auto& col : out.feat_cols) col = r.read_str();
    uint32_t n_med = r.read<uint32_t>();
    if (r.fail || n_med > 100000) return false;
    for (uint32_t i = 0; i < n_med; i++) {
        std::string col = r.read_str();
        if (r.fail || r.pos + 8 > r.size) return false;
        double val; std::memcpy(&val, r.data + r.pos, 8); r.pos += 8;
        out.medians[col] = val;
    }
    return r.ok();
}

// ── 推論 ─────────────────────────────────────────────────────────────────────
// predict() が y_transform 逆変換を(blendメンバーも含め再帰的に)適用するため、
// 定義(Y逆変換セクション、このファイルの後方)より前方宣言しておく。
static float inv_ytransform(float pred, YTransform yt, float lam);

static float predict_linear(const LinearModel& m, const float* x, int d) {
    float s = m.intercept;
    for (int i = 0; i < d; i++) {
        float sc = (x[i] - m.mean[i]) / (m.scale[i] + 1e-8f);
        s += m.coef[i] * sc;
    }
    return s;
}

static float predict_lgbm(const LGBMModel& m, const float* x) {
    double sum = 0.0;
    for (const auto& tree : m.trees) {
        if (tree.n_leaves == 1) {  // 単葉ツリー: 定数寄与
            sum += tree.leaf_value[0];
            continue;
        }
        int node = 0;
        while (true) {
            int feat = (int)tree.split_feature[node];
            float thr = tree.threshold[node];
            int next = (x[feat] <= thr) ? tree.left_child[node] : tree.right_child[node];
            if (next < 0) { sum += tree.leaf_value[-(next + 1)]; break; }
            node = next;
        }
    }
    return (float)sum;
}

static float predict_gp(const GPModel& m, const float* x) {
    const int d = m.n_feat;
    std::vector<float> xs(d);
    for (int i = 0; i < d; i++)
        xs[i] = (x[i] - m.mean[i]) / (m.scale[i] + 1e-8f);

    double y_norm = 0.0;
    for (int i = 0; i < m.n_train; i++) {
        double sq = 0.0;
        for (int j = 0; j < d; j++) {
            double diff = xs[j] - m.X_train[i * d + j];
            double ls_j = m.ls[j];
            sq += (diff / ls_j) * (diff / ls_j);
        }
        y_norm += m.sv * std::exp(-0.5 * sq) * m.alpha[i];
    }
    return (float)(y_norm * m.y_std + m.y_mean);
}

static float predict_mlp(const MLPModel& m, const float* x, int d) {
    std::vector<float> h(d);
    for (int i = 0; i < d; i++)
        h[i] = (x[i] - m.mean[i]) / (m.scale[i] + 1e-8f);

    for (const auto& layer : m.layers) {
        std::vector<float> out_v(layer.n_out, 0.0f);
        // sklearn: W shape = (n_in, n_out), row-major → W[k*n_out + j]
        for (int j = 0; j < layer.n_out; j++) {
            float val = layer.b[j];
            for (int k = 0; k < layer.n_in; k++)
                val += h[k] * layer.W[k * layer.n_out + j];
            out_v[j] = val;
        }
        if (layer.act == 0) {  // relu
            for (auto& v : out_v) v = std::max(0.0f, v);
        }
        h = std::move(out_v);
    }
    return h.empty() ? 0.0f : h[0];
}

// linear_poly (多項式Ridge): 標準化 → (単項 or 標準化後の値どうしの積/二乗) →
// coef内積 + intercept。term_b<0 は単項(s[term_a]そのまま)を表す
// (train_bridge._write_treg_stream / predict-core.js predictLinearPoly と同一ロジック)。
static float predict_linear_poly(const LinearPolyModel& m, const float* x, int d) {
    std::vector<float> s(d);
    for (int i = 0; i < d; i++)
        s[i] = (x[i] - m.center[i]) / (m.scale[i] + 1e-8f);

    double sum = m.intercept;
    const size_t n = m.coef.size();
    for (size_t t = 0; t < n; t++) {
        int32_t a = m.term_a[t], b = m.term_b[t];
        float val = (b < 0) ? s[a] : s[a] * s[b];
        sum += (double)m.coef[t] * val;
    }
    return (float)sum;
}

// v4 派生特徴: ソース列を（学習時と同じ境界で）クリップして取得。欠損は NaN。
static double clipped_source(const std::map<std::string, double>& row,
                             const std::string& col, float lo, float hi) {
    auto it = row.find(col);
    if (it == row.end() || std::isnan(it->second))
        return std::numeric_limits<double>::quiet_NaN();
    return std::min(std::max(it->second, (double)lo), (double)hi);
}

// 派生特徴を計算する。ソース欠損・非有限は NaN（→ 呼び出し側で median 補完）。
static double compute_derived(const DerivedFeat& df, const std::map<std::string, double>& row) {
    double a = clipped_source(row, df.col_a, df.a_lo, df.a_hi);
    if (std::isnan(a)) return a;
    double v;
    switch (df.op) {
        case DOP_MUL: {
            double b = clipped_source(row, df.col_b, df.b_lo, df.b_hi);
            if (std::isnan(b)) return b;
            v = a * b;
            break;
        }
        case DOP_SQ:   v = a * a; break;
        case DOP_SIGN: v = (double)((a > 0.0) - (a < 0.0)); break;
        default: return std::numeric_limits<double>::quiet_NaN();
    }
    return std::isfinite(v) ? v : std::numeric_limits<double>::quiet_NaN();
}

// blend(アンサンブル)のメンバーは後処理なし(smear=1, y_clip=無制限, round無し)の
// 自己完結した .treg として書き出されているため、predict() をメンバーごとに再帰
// 呼び出しして得た「実スケールの予測」(＝各メンバー自身のy_transform逆変換済み)を
// 重み付き和するだけでよい。最終的な smear/y_clip/round_output は呼び出し元の
// predict()(外側のblendモデル自身)側で一度だけ適用される
// (JS版 predict-core.js の predictBlend/predictRow と同一設計)。
static float predict(const TregModel& model, const std::map<std::string, double>& row);

static float predict_blend(const BlendModel& m, const std::map<std::string, double>& row) {
    double sum = 0.0;
    for (const auto& mem : m.members)
        sum += (double)mem.weight * predict(*mem.model, row);
    return (float)sum;
}

static float predict(const TregModel& model, const std::map<std::string, double>& row) {
    float pred;
    if (model.type == MT_BLEND) {
        // blendは自身の直接の特徴ベクトルを持たない(各メンバーが個別に持つ)ため、
        // ここでは x を組み立てない。
        pred = predict_blend(model.blend, row);
    } else {
        const int d = (int)model.feat_cols.size();
        std::vector<float> x(d);
        for (int i = 0; i < d; i++) {
            double val = std::numeric_limits<double>::quiet_NaN();
            auto dit = model.derived_idx.find(model.feat_cols[i]);
            if (dit != model.derived_idx.end()) {
                val = compute_derived(model.derived[dit->second], row);
            } else {
                auto it = row.find(model.feat_cols[i]);
                if (it != row.end()) val = it->second;
            }
            if (!std::isnan(val)) {
                x[i] = (float)val;
            } else {
                auto mit = model.medians.find(model.feat_cols[i]);
                x[i] = (mit != model.medians.end()) ? (float)mit->second : 0.0f;
            }
            // 学習時と同じ X クリッピング（1-99パーセンタイル）を適用
            if (i < (int)model.x_clip_lo.size())
                x[i] = std::min(std::max(x[i], model.x_clip_lo[i]), model.x_clip_hi[i]);
        }
        switch (model.type) {
            case MT_LINEAR:      pred = predict_linear(model.linear, x.data(), d); break;
            case MT_LGBM:        pred = predict_lgbm(model.lgbm, x.data()); break;
            case MT_GP:          pred = predict_gp(model.gp, x.data()); break;
            case MT_MLP:         pred = predict_mlp(model.mlp, x.data(), d); break;
            case MT_LINEAR_POLY: pred = predict_linear_poly(model.linear_poly, x.data(), d); break;
            default:             pred = 0.0f; break;
        }
    }
    // Y 逆変換 (log1p / Yeo-Johnson) + 予測後処理 (smearing補正 / Y観測レンジclip / 整数丸め)。
    // blendの外側モデルは y_transform=NONE で書かれる(各メンバーで既に逆変換済みのため
    // 二重変換を避ける、train_bridge._write_treg_stream 参照)ので、ここでの適用は
    // メンバーごとに個別・外側で一度、という2段構成が自然に成立する。
    pred = inv_ytransform(pred, model.y_transform, model.yeo_lambda);
    pred *= model.smear;
    pred = std::min(std::max(pred, model.y_clip_lo), model.y_clip_hi);
    if (model.round_output) pred = std::round(pred);
    return pred;
}

// ── Y 逆変換 ─────────────────────────────────────────────────────────────────
static float yeo_johnson_inv(float y, float lam) {
    if (y >= 0.0f) {
        if (std::abs(lam) < 1e-6f) return std::expm1(y);
        return std::pow(lam * y + 1.0f, 1.0f / lam) - 1.0f;
    } else {
        float lam2 = 2.0f - lam;
        if (std::abs(lam2) < 1e-6f) return 1.0f - std::exp(-y);
        return 1.0f - std::pow(-lam2 * y + 1.0f, 1.0f / lam2);
    }
}

static float inv_ytransform(float pred, YTransform yt, float lam) {
    switch (yt) {
        case YT_LOG1P:       return std::expm1(pred);
        case YT_YEO_JOHNSON: return yeo_johnson_inv(pred, lam);
        default:             return pred;
    }
}

// ── EXE テール自己埋め込み読み込み ───────────────────────────────────────────
static std::vector<uint8_t> load_embedded_treg(const char* exe_path) {
    FILE* f = std::fopen(exe_path, "rb");
    if (!f) return {};
    std::fseek(f, 0, SEEK_END);
    long total = std::ftell(f);
    if (total < 16) { std::fclose(f); return {}; }

    char magic[8];
    std::fseek(f, total - 8, SEEK_SET);
    std::fread(magic, 1, 8, f);
    if (std::memcmp(magic, EXE_MAGIC, 8) != 0) { std::fclose(f); return {}; }

    uint64_t treg_size;
    std::fseek(f, total - 16, SEEK_SET);
    std::fread(&treg_size, 8, 1, f);
    if ((long)treg_size > total - 16) { std::fclose(f); return {}; }

    std::vector<uint8_t> buf(treg_size);
    std::fseek(f, total - 16 - (long)treg_size, SEEK_SET);
    std::fread(buf.data(), 1, treg_size, f);
    std::fclose(f);
    return buf;
}

// ── CSV パーサー ──────────────────────────────────────────────────────────────
struct CsvData {
    std::vector<std::string> headers;
    std::vector<std::map<std::string, double>> rows;
};

static std::string trim_csv_field(std::string s) {
    // CR, LF のみを外側から除去する。
    // 旧実装は「先頭バイトが0xEF/0xBB/0xBFのいずれかなら1バイト削る」を全フィールドに
    // 適用しており、半角カナ・全角英数字などUTF-8の1バイト目がたまたま0xEFと一致する
    // 文字（例: "ｺｽﾄ"=EF BD BA、"１号機"=EF BC 91）の先頭バイトを誤ってBOMの残骸と
    // 見なして剥ぎ取り、ヘッダ名が破損してfeat_cols不一致→全行median予測になっていた
    // (中-M5)。BOM除去はヘッダ行全体に対して1回だけ、3バイト連続一致時のみ行う
    // (parse_csv側で実施)。
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n'))
        s.pop_back();
    // 前後の空白(半角スペース・タブ)をトリムする。pandas側(df.columns.str.strip()、
    // 中-M7)・フロントエンド(frontend/index.html parseHeaderLine の .trim()相当)・
    // lib.rs(read_csv_headers)はいずれも空白をトリムするため、ここで揃えないと
    // 「x1, y」のようなヘッダを持つCSVで native exe だけ " y" のまま残り、
    // feat_colsと不一致→サイレントmedian予測になる(独立検証で発見の残余非対称)。
    auto is_ws = [](unsigned char c) { return c == ' ' || c == '\t'; };
    size_t start = 0, end = s.size();
    while (start < end && is_ws((unsigned char)s[start])) start++;
    while (end > start && is_ws((unsigned char)s[end - 1])) end--;
    s = s.substr(start, end - start);
    // クォート除去も先頭 " または ' を無条件に1文字ずつ剥いでいたため、'開始不一致'な
    // フィールド（引用符で囲まれていない、末尾にアポストロフィを含むだけの通常文字列）
    // まで壊していた。ダブルクォートで前後を囲まれている場合のみペアで除去する。
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        s = s.substr(1, s.size() - 2);
    }
    return s;
}

// ファイル冒頭のUTF-8 BOM(EF BB BF)を、3バイト連続一致時のみ1回だけ除去する。
// trim_csv_fieldのように個々のバイト値だけで判定すると、半角カナ・全角文字の
// UTF-8エンコード1バイト目(0xEF)を誤ってBOMと誤認してしまう(中-M5)。
static std::string strip_utf8_bom_once(const std::string& s) {
    if (s.size() >= 3 && (unsigned char)s[0] == 0xEF &&
        (unsigned char)s[1] == 0xBB && (unsigned char)s[2] == 0xBF)
        return s.substr(3);
    return s;
}

static CsvData parse_csv(const std::string& path) {
    CsvData result;
    std::ifstream file(path);
    std::string line;
    if (!std::getline(file, line)) return result;
    line = strip_utf8_bom_once(line);

    std::stringstream hss(line);
    std::string col;
    while (std::getline(hss, col, ','))
        result.headers.push_back(trim_csv_field(col));

    while (std::getline(file, line)) {
        if (line.empty() || line == "\r") continue;
        std::map<std::string, double> row;
        std::stringstream rss(line);
        std::string val;
        int idx = 0;
        while (std::getline(rss, val, ',') && idx < (int)result.headers.size()) {
            val = trim_csv_field(val);
            // std::stod は "12abc" のような末尾に余分な文字がある文字列でも先頭の
            // "12" だけを解釈して例外を投げずに成功してしまう。Web版JS(Number()ベースの
            // 全体一致)・Python版(pd.to_numeric)は同じセルをNaN(欠損→median補完)として
            // 扱うため、native版だけ不正な数値プレフィックスをそのまま使ってしまい
            // 挙動が食い違っていた(中-10)。posを見て文字列全体が数値として消費されて
            // いない場合はNaN扱いにし、他の実装と揃える。
            try {
                size_t consumed = 0;
                double v = std::stod(val, &consumed);
                while (consumed < val.size() && std::isspace((unsigned char)val[consumed])) consumed++;
                row[result.headers[idx]] = (consumed == val.size())
                    ? v : std::numeric_limits<double>::quiet_NaN();
            } catch (...) {
                row[result.headers[idx]] = std::numeric_limits<double>::quiet_NaN();
            }
            idx++;
        }
        result.rows.push_back(row);
    }
    return result;
}

// ── main ─────────────────────────────────────────────────────────────────────
// 実体は run() に置き、main() はそれを try/catch で包むだけにする(低-M12)。
// 破損した.treg(境界検査をすり抜けた巨大なn_fc_clip等)がstd::bad_alloc/length_errorを
// 送出すると、以前は未捕捉のままabort()していた(exit=134、-mwindowsビルドでは
// メッセージも出ず無言クラッシュ)。ここで捕捉しfatal()でユーザーに知らせる。
static int run(int argc, char* argv[]) {
    if (argc < 2) {
        fatal("CSVファイルをこのEXEにドラッグ＆ドロップしてください。");
        return 1;
    }
    const char* csv_path = argv[1];

    // .treg を読み込む: 1) 自己埋め込み  2) argv[2]  3) exe 隣の trained_model/model.treg
    TregModel model;
    bool loaded = false;

    auto embedded = load_embedded_treg(argv[0]);
    if (!embedded.empty()) {
        loaded = load_treg(embedded.data(), embedded.size(), model);
    }

    if (!loaded) {
        std::string treg_path;
        if (argc >= 3) {
            treg_path = argv[2];
        } else {
            std::string exe(argv[0]);
            auto sep = exe.find_last_of("/\\");
            std::string dir = (sep != std::string::npos) ? exe.substr(0, sep + 1) : "./";
            treg_path = dir + "trained_model/model.treg";
        }
        FILE* f = std::fopen(treg_path.c_str(), "rb");
        if (!f) {
            std::string msg = "モデルファイルが見つかりません:\n" + treg_path;
            fatal(msg.c_str());
            return 1;
        }
        std::fseek(f, 0, SEEK_END);
        long sz = std::ftell(f);
        std::rewind(f);
        std::vector<uint8_t> buf(sz);
        std::fread(buf.data(), 1, sz, f);
        std::fclose(f);
        loaded = load_treg(buf.data(), buf.size(), model);
    }

    if (!loaded) {
        fatal("モデルの読み込みに失敗しました。");
        return 1;
    }

    // CSV 解析
    CsvData csv = parse_csv(csv_path);
    if (csv.headers.empty()) {
        fatal("CSVファイルを読み込めませんでした。\nファイルが壊れていないか確認してください。");
        return 1;
    }

    // 欠損補完して予測(Y逆変換・smearing補正・観測レンジclip・整数丸めは predict() 内で
    // 適用済み。blendの場合はメンバーごとの逆変換 → 外側の後処理、の順で再帰的に行われる)。
    std::vector<float> preds;
    preds.reserve(csv.rows.size());
    for (const auto& row : csv.rows)
        preds.push_back(predict(model, row));

    // 出力 CSV 作成: 入力 CSV と同じディレクトリに {stem}_pred.csv
    std::string csv_dir, csv_stem;
    {
        std::string csv_str(csv_path);
        auto sep = csv_str.find_last_of("/\\");
        if (sep != std::string::npos) {
            csv_dir  = csv_str.substr(0, sep + 1);
            csv_stem = csv_str.substr(sep + 1);
        } else {
            csv_dir  = "./";
            csv_stem = csv_str;
        }
        auto dot = csv_stem.rfind('.');
        if (dot != std::string::npos) csv_stem = csv_stem.substr(0, dot);
    }
    std::string out_path = csv_dir + csv_stem + "_pred.csv";

    std::ifstream in_file(csv_path);
    std::ofstream out_file(out_path);
    if (!out_file.is_open()) {
        std::string msg = "結果ファイルを作成できませんでした:\n" + out_path;
        fatal(msg.c_str());
        return 1;
    }

    // ostream既定のprecision(6)のままだと百万円台以上の値が"1.23457e+06"のように
    // 有効桁6桁+指数表記へ丸められ、round_output=trueでも印字段階で非整数化して
    // 誤差が出る(中-M6)。float の全桁を再現できる max_digits10=9 を設定する。
    out_file << std::setprecision(9);

    std::string header_line;
    std::getline(in_file, header_line);
    while (!header_line.empty() && (header_line.back() == '\r' || header_line.back() == '\n'))
        header_line.pop_back();
    out_file << header_line << "," << model.target_col << "\n";

    int row_idx = 0;
    std::string row_line;
    while (std::getline(in_file, row_line)) {
        if (row_line.empty() || row_line == "\r") continue;
        while (!row_line.empty() && (row_line.back() == '\r' || row_line.back() == '\n'))
            row_line.pop_back();
        float pred = (row_idx < (int)preds.size()) ? preds[row_idx] : 0.0f;
        out_file << row_line << "," << pred << "\n";
        row_idx++;
    }
    out_file.close();

    // 完了通知は表示しない。結果はカレントフォルダ（CSVと同じ場所）へ静かに出力する。
    return 0;
}

int main(int argc, char* argv[]) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::string msg = std::string("予期しないエラーが発生しました:\n") + e.what() +
                           "\n\nモデルファイルまたはCSVが壊れている可能性があります。";
        fatal(msg.c_str());
        return 1;
    } catch (...) {
        fatal("予期しないエラーが発生しました。\nモデルファイルまたはCSVが壊れている可能性があります。");
        return 1;
    }
}
