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
 * ビルド (MinGW, Windows。CP932外文字を含むパスのD&Dに対応するため GetCommandLineW+
 *         CommandLineToArgvW を使う。shell32 のリンクと、_wfopen 等の非標準拡張を
 *         見えるようにするための -std=gnu++17 が必要):
 *   g++ -O2 -std=gnu++17 -static -mwindows -s predict_native_v2.cpp -lshell32 -o predict_native.exe
 * ビルド (CI/ubuntu、非Windows。通常の main・fopen を使用):
 *   g++ -O2 -std=c++17 predict_native_v2.cpp -o predict_native_ref
 * ビルド (MSVC):
 *   cl /O2 /EHsc /std:c++17 predict_native_v2.cpp /Fe:predict_native.exe
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cctype>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <limits>
#include <memory>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>  // CommandLineToArgvW (中-M2)
static void fatal(const char* msg) {
    // テスト容易性フック: MessageBoxW はモーダルダイアログで、クリックする人間がいない
    // 自動テスト(壊れた.tregを与えてクラッシュせずエラー終了することを確認するスクリプト
    // 等)ではプロセスが無期限にハングしてしまう。環境変数 TREG_NO_GUI が設定されている
    // 場合のみ、GUIダイアログの代わりに stderr へ出力する(本番の配布exeの挙動は不変)。
    if (std::getenv("TREG_NO_GUI")) {
        std::fprintf(stderr, "ERROR: %s\n", msg);
        return;
    }
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
enum DerivedOp  { DOP_MUL = 0, DOP_SQ = 1, DOP_SIGN = 2 };

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
// depth: blend入れ子の再帰段数。現行の writer(train_bridge._write_treg_stream)は
// blendメンバーを常に「非blendの自己完結モデル」として書き出し、blendの入れ子(blend
// の中にblend)は生成しない。1MB程度の細工された.tregでも depth>1 を拒否しないと
// 数万段の再帰でC++スタックオーバーフローを起こせてしまう(中-M1)ため、ここで
// 明示的に上限を課す。
static bool load_treg(const uint8_t* data, size_t size, TregModel& out, int depth = 0) {
    if (depth > 1) return false;  // blendの入れ子は1段まで(現行フォーマットで正当なのは1段)

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
            // 整合性検査(重大-1): split_feature は特徴量次元内、left/right_child は
            // 「葉」(負値、-n_leaves以上)か「内部ノード」(0以上ni未満)のいずれかで
            // なければならない。これを検査しないと predict_lgbm() が x[feat] や
            // tree.left_child[node] でヒープ範囲外読み出し(UB)を起こしうる、
            // または循環参照で無限ループしうる(後者は推論ループ側の訪問回数上限でも防ぐ)。
            for (int i = 0; i < ni; i++) {
                if (tree.split_feature[i] >= (uint32_t)d) return false;
                if (tree.left_child[i]  < -(int32_t)tree.n_leaves || tree.left_child[i]  >= ni) return false;
                if (tree.right_child[i] < -(int32_t)tree.n_leaves || tree.right_child[i] >= ni) return false;
            }
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
        // 整合性検査(重大-1): 層の次元チェーンが破綻していると predict_mlp() の
        // W[k*n_out+j] がヒープ範囲外読み出しになる。
        if (out.mlp.layers.empty() || out.mlp.layers[0].n_in != d) return false;
        for (size_t li = 1; li < out.mlp.layers.size(); li++) {
            if (out.mlp.layers[li].n_in != out.mlp.layers[li - 1].n_out) return false;
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
            if (!load_treg(r.data + r.pos, blob_len, *member, depth + 1)) return false;
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
    // 整合性検査(重大-1): ヘッダの n_feat とテールの n_fc(feat_cols実個数)が食い違う
    // 破損ファイルを拒否する。一致していないと predict() が feat_cols を n_feat 個
    // 前提で読む一方、実際のスケーラ/係数配列は元の n_feat 個ぶんしか無い(あるいは逆)
    // というズレが起き、範囲外読み出しにつながる。
    if ((uint32_t)out.n_feat != n_fc) return false;
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

// スケーラのεは.treg書き出し時に焼き込み済み(train_bridge._write_treg_stream が
// scale=max(scale,1e-8)にしてから float32 化する。中-M3)。読み込み側は「.tregの値で
// 割るだけ」でよく、以前のように読み込み側で +1e-8 を足す必要はない(JS版
// predict-core.js / predict_template.html インライン版と同一の意味論に統一)。
static float predict_linear(const LinearModel& m, const float* x, int d) {
    double s = m.intercept;
    for (int i = 0; i < d; i++) {
        double sc = ((double)x[i] - (double)m.mean[i]) / (double)m.scale[i];
        s += (double)m.coef[i] * sc;
    }
    return (float)s;
}

static float predict_lgbm(const LGBMModel& m, const float* x) {
    double sum = 0.0;
    for (const auto& tree : m.trees) {
        if (tree.n_leaves == 1) {  // 単葉ツリー: 定数寄与
            sum += tree.leaf_value[0];
            continue;
        }
        int node = 0;
        // 循環参照(細工された.treg)による無限ループを防ぐため、訪問回数に上限を
        // 設ける(重大-1)。load_treg 側の範囲検査で split_feature/left_child/right_child
        // の値域は保証済みだが、値域内でも「サイクルを構成する」ことは静的検査だけでは
        // 防ぎきれないため、実行時にも二重の防御をかける。
        const size_t max_visits = tree.split_feature.size();  // = 内部ノード数(ni)
        size_t visits = 0;
        while (true) {
            if (++visits > max_visits)
                throw std::runtime_error("LGBM木の巡回上限を超過しました(破損した.tregの可能性があります)");
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
        xs[i] = (x[i] - m.mean[i]) / m.scale[i];

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
        h[i] = (float)(((double)x[i] - (double)m.mean[i]) / (double)m.scale[i]);

    for (const auto& layer : m.layers) {
        std::vector<float> out_v(layer.n_out, 0.0f);
        // sklearn: W shape = (n_in, n_out), row-major → W[k*n_out + j]
        // 中-M7: 累積は double で行い、層をまたぐ受け渡し(out_v への格納)時にのみ
        // float32 へ落とす。JS版(Float32Array へ格納する箇所でのみ丸められる)と
        // 丸めのタイミングを揃えるための変更(以前はCのみ全経路float精度だった)。
        for (int j = 0; j < layer.n_out; j++) {
            double val = (double)layer.b[j];
            for (int k = 0; k < layer.n_in; k++)
                val += (double)h[k] * (double)layer.W[k * layer.n_out + j];
            out_v[j] = (float)val;
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
        s[i] = (x[i] - m.center[i]) / m.scale[i];

    double sum = m.intercept;
    const size_t n = m.coef.size();
    for (size_t t = 0; t < n; t++) {
        int32_t a = m.term_a[t], b = m.term_b[t];
        float val = (b < 0) ? s[a] : s[a] * s[b];
        sum += (double)m.coef[t] * val;
    }
    return (float)sum;
}

// 低-M4: 行データは(以前の)行ごと std::map<std::string,double> ではなく、CSVヘッダの
// 列順に整列した std::vector<double> として保持する。列名→インデックスの対応
// (header_idx)は CSV 全体で1回だけ構築し、行ごとの再ハッシュ/再アロケーションを
// なくす(巨大CSVでのメモリ・速度改善)。
using HeaderIndex = std::unordered_map<std::string, size_t>;

static double lookup_col(const std::vector<double>& row_vals, const HeaderIndex& header_idx,
                         const std::string& col) {
    auto it = header_idx.find(col);
    if (it == header_idx.end()) return std::numeric_limits<double>::quiet_NaN();
    return row_vals[it->second];
}

// v4 派生特徴: ソース列を（学習時と同じ境界で）クリップして取得。欠損は NaN。
static double clipped_source(const std::vector<double>& row_vals, const HeaderIndex& header_idx,
                             const std::string& col, float lo, float hi) {
    double v = lookup_col(row_vals, header_idx, col);
    if (std::isnan(v)) return v;
    return std::min(std::max(v, (double)lo), (double)hi);
}

// 派生特徴を計算する。ソース欠損・非有限は NaN（→ 呼び出し側で median 補完）。
static double compute_derived(const DerivedFeat& df, const std::vector<double>& row_vals,
                              const HeaderIndex& header_idx) {
    double a = clipped_source(row_vals, header_idx, df.col_a, df.a_lo, df.a_hi);
    if (std::isnan(a)) return a;
    double v;
    switch (df.op) {
        case DOP_MUL: {
            double b = clipped_source(row_vals, header_idx, df.col_b, df.b_lo, df.b_hi);
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
static float predict(const TregModel& model, const std::vector<double>& row_vals,
                     const HeaderIndex& header_idx);

static float predict_blend(const BlendModel& m, const std::vector<double>& row_vals,
                           const HeaderIndex& header_idx) {
    double sum = 0.0;
    for (const auto& mem : m.members)
        sum += (double)mem.weight * predict(*mem.model, row_vals, header_idx);
    return (float)sum;
}

static float predict(const TregModel& model, const std::vector<double>& row_vals,
                     const HeaderIndex& header_idx) {
    float pred;
    if (model.type == MT_BLEND) {
        // blendは自身の直接の特徴ベクトルを持たない(各メンバーが個別に持つ)ため、
        // ここでは x を組み立てない。
        pred = predict_blend(model.blend, row_vals, header_idx);
    } else {
        const int d = (int)model.feat_cols.size();
        std::vector<float> x(d);
        for (int i = 0; i < d; i++) {
            double val = std::numeric_limits<double>::quiet_NaN();
            auto dit = model.derived_idx.find(model.feat_cols[i]);
            if (dit != model.derived_idx.end()) {
                val = compute_derived(model.derived[dit->second], row_vals, header_idx);
            } else {
                val = lookup_col(row_vals, header_idx, model.feat_cols[i]);
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

// 予測に必要な「生」列(派生特徴はソース列に展開、blendは全メンバーのunion)を集める。
// 中-M4b: 以前(predict_template.htmlの旧rawRequiredColumns相当)はblendの外側モデルの
// feat_cols([]、blend自身は特徴を持たないため)しか見ておらず、blendでは列欠損警告が
// 絶対に出なかった。メンバー再帰でunionする。
static void collect_required_raw_columns(const TregModel& model, std::vector<std::string>& out) {
    if (model.type == MT_BLEND) {
        for (auto& mem : model.blend.members) collect_required_raw_columns(*mem.model, out);
        return;
    }
    for (auto& name : model.feat_cols) {
        auto dit = model.derived_idx.find(name);
        if (dit != model.derived_idx.end()) {
            const auto& df = model.derived[dit->second];
            out.push_back(df.col_a);
            if (df.op == DOP_MUL) out.push_back(df.col_b);
        } else {
            out.push_back(name);
        }
    }
}

static std::vector<std::string> required_raw_columns_unique(const TregModel& model) {
    std::vector<std::string> raw;
    collect_required_raw_columns(model, raw);
    std::vector<std::string> uniq;
    std::unordered_set<std::string> seen;
    for (auto& c : raw) if (seen.insert(c).second) uniq.push_back(c);
    return uniq;
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

// ── Windows向けユーティリティ(UTF-8パス⇔ワイド文字、cp932→UTF-8変換) ────────────
// 中-M2: 従来 argv[0]/fopen(narrow) 依存だったため、(1) exe自身のパス取得に
// argv[0]という「呼び出し側が渡した値」に頼っていた、(2) narrow版 fopen/ifstream は
// Windows では現在のANSIコードページでパスをデコードするため、cp932の範囲外の文字
// (絵文字・一部の外国語文字等)を含むパスはそもそも開けなかった。wmain + ワイド文字
// API(_wfopen/GetModuleFileNameW)に統一することでこれを解消する(中-M2)。
#ifdef _WIN32
static std::wstring utf8_to_wide(const std::string& s) {
    if (s.empty()) return std::wstring();
    int wlen = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), NULL, 0);
    if (wlen <= 0) return std::wstring();
    std::wstring w(wlen, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), &w[0], wlen);
    return w;
}
static std::string wide_to_utf8(const wchar_t* w, int wlen /* -1 ならNUL終端文字列 */) {
    int ulen = WideCharToMultiByte(CP_UTF8, 0, w, wlen, NULL, 0, NULL, NULL);
    if (ulen <= 0) return std::string();
    std::string u(ulen, 0);
    WideCharToMultiByte(CP_UTF8, 0, w, wlen, &u[0], ulen, NULL, NULL);
    if (wlen < 0 && !u.empty() && u.back() == '\0') u.pop_back();  // NUL終端ぶんを除去
    return u;
}
static FILE* fopen_utf8(const std::string& path_utf8, const char* mode) {
    std::wstring wp = utf8_to_wide(path_utf8);
    std::wstring wm(mode, mode + std::strlen(mode));
    return _wfopen(wp.c_str(), wm.c_str());
}
// exe自身の実パスをOSに問い合わせる(argv[0]はシェル/呼び出し元がどう渡すか次第で
// 相対パス・不完全なパスのことがあるため、GetModuleFileNameW の方が確実)。
static std::string get_exe_path_utf8() {
    std::vector<wchar_t> buf(1024);
    for (;;) {
        DWORD n = GetModuleFileNameW(NULL, buf.data(), (DWORD)buf.size());
        if (n == 0) return std::string();
        if (n < buf.size() - 1) return wide_to_utf8(buf.data(), (int)n);
        if (buf.size() > 65536) return std::string();  // 異常に長いパスは諦める
        buf.resize(buf.size() * 2);
    }
}
// EXEテール読み出し用: ftell の long(Windowsでは32bit)は2GB超のEXEで破綻するため
// 64bit版を使う(中-M2)。
using file_off_t = long long;
#define FTELL _ftelli64
#define FSEEK _fseeki64
#else
static FILE* fopen_utf8(const std::string& path_utf8, const char* mode) {
    return std::fopen(path_utf8.c_str(), mode);
}
using file_off_t = long;
#define FTELL std::ftell
#define FSEEK std::fseek
#endif

// ファイル全体をバイト列として読み込む(CSV読み込み・EXEテール自己埋め込み.treg
// 読み込みの両方で共用)。fseek/freadの戻り値を検査し、途中で読み取りに失敗したら
// 空/失敗として扱う(中-M2: 以前は戻り値未検査だった)。
static std::string read_whole_file(const std::string& path, bool& ok) {
    ok = false;
    FILE* f = fopen_utf8(path, "rb");
    if (!f) return {};
    if (FSEEK(f, 0, SEEK_END) != 0) { std::fclose(f); return {}; }
    file_off_t sz = FTELL(f);
    if (sz < 0) { std::fclose(f); return {}; }
    if (FSEEK(f, 0, SEEK_SET) != 0) { std::fclose(f); return {}; }
    std::string buf;
    buf.resize((size_t)sz);
    if (sz > 0 && std::fread(&buf[0], 1, (size_t)sz, f) != (size_t)sz) {
        std::fclose(f); return {};
    }
    std::fclose(f);
    ok = true;
    return buf;
}

// ── EXE テール自己埋め込み読み込み ───────────────────────────────────────────
static std::vector<uint8_t> load_embedded_treg(const std::string& exe_path) {
    FILE* f = fopen_utf8(exe_path, "rb");
    if (!f) return {};
    if (FSEEK(f, 0, SEEK_END) != 0) { std::fclose(f); return {}; }
    file_off_t total = FTELL(f);
    if (total < 16) { std::fclose(f); return {}; }

    char magic[8];
    if (FSEEK(f, total - 8, SEEK_SET) != 0 || std::fread(magic, 1, 8, f) != 8) {
        std::fclose(f); return {};
    }
    if (std::memcmp(magic, EXE_MAGIC, 8) != 0) { std::fclose(f); return {}; }

    uint64_t treg_size;
    if (FSEEK(f, total - 16, SEEK_SET) != 0 || std::fread(&treg_size, 8, 1, f) != 1) {
        std::fclose(f); return {};
    }
    // 低-M2: 以前は treg_size(uint64_t) を (long) に縮小キャストしてから比較しており、
    // Windows(long=32bit)では 4GB 境界で符号/桁が壊れ、本来失敗すべき巨大な値が
    // チェックをすり抜けうった。縮小キャストせず符号なしのまま比較する。
    if (treg_size > (uint64_t)(total - 16)) { std::fclose(f); return {}; }

    std::vector<uint8_t> buf(treg_size);
    if (!buf.empty()) {
        if (FSEEK(f, total - 16 - (file_off_t)treg_size, SEEK_SET) != 0 ||
            std::fread(buf.data(), 1, treg_size, f) != treg_size) {
            std::fclose(f); return {};
        }
    }
    std::fclose(f);
    return buf;
}

// ── CSV パーサー ──────────────────────────────────────────────────────────────
struct CsvData {
    std::vector<std::string> headers;
    std::vector<std::vector<std::string>> rows;  // 生フィールド(クォート・CR/LF除去済み)
};

static std::string trim_ws(const std::string& s) {
    auto is_ws = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
    size_t start = 0, end = s.size();
    while (start < end && is_ws((unsigned char)s[start])) start++;
    while (end > start && is_ws((unsigned char)s[end - 1])) end--;
    return s.substr(start, end - start);
}

// 高-2: 単純カンマ分割をやめ、web/predict_template.html の parseCSV と同一の状態機械
// (inQuotes、""エスケープ、引用符内改行・カンマ)に置き換える。以前はExcelの
// `"a,b"` のようなクォート付きセルで列がずれ、フィールドの対応が崩れたまま
// median補完でサイレントに誤予測していた。ファイル全体を一度に読んでトークナイズする
// ため、引用符内改行をまたぐフィールドも正しく1行として扱える(行単位のgetlineでは
// 不可能だった)。
static std::vector<std::vector<std::string>> tokenize_csv(const std::string& text) {
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> row;
    std::string field;
    bool in_quotes = false;
    size_t i = 0, n = text.size();
    while (i < n) {
        char c = text[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < n && text[i + 1] == '"') { field += '"'; i += 2; continue; }
                in_quotes = false; i++; continue;
            }
            field += c; i++; continue;
        }
        if (c == '"') { in_quotes = true; i++; continue; }
        if (c == ',') { row.push_back(field); field.clear(); i++; continue; }
        if (c == '\r') { i++; continue; }
        if (c == '\n') { row.push_back(field); rows.push_back(row); row.clear(); field.clear(); i++; continue; }
        field += c; i++;
    }
    if (!field.empty() || !row.empty()) { row.push_back(field); rows.push_back(row); }
    while (!rows.empty() && rows.back().size() == 1 && rows.back()[0].empty()) rows.pop_back();
    return rows;
}

// UTF-8として妥当か(継続バイト構造)を検査する。ヘッダ行のみに適用する軽量な検査で、
// 完全なUTF-8バリデータではないが cp932 由来の不正な継続バイト列は確実に検出できる。
static bool is_valid_utf8(const char* s, size_t len) {
    size_t i = 0;
    while (i < len) {
        unsigned char c = (unsigned char)s[i];
        if (c < 0x80) { i++; continue; }
        int extra;
        if ((c & 0xE0) == 0xC0) extra = 1;
        else if ((c & 0xF0) == 0xE0) extra = 2;
        else if ((c & 0xF8) == 0xF0) extra = 3;
        else return false;
        if (i + extra >= len) return false;
        for (int k = 1; k <= extra; k++) {
            unsigned char cc = (unsigned char)s[i + k];
            if ((cc & 0xC0) != 0x80) return false;
        }
        i += (size_t)extra + 1;
    }
    return true;
}

#ifdef _WIN32
// 高-3: ヘッダがUTF-8として不正(日本語Excel既定のcp932/Shift-JISの可能性が高い)な
// 場合、Windows では MultiByteToWideChar(932)→WideCharToMultiByte(CP_UTF8) で
// cp932→UTF-8変換を試みる(Web版frontend/index.htmlの同種フォールバックと方針を揃える)。
static std::string cp932_to_utf8(const std::string& s) {
    int wlen = MultiByteToWideChar(932, 0, s.data(), (int)s.size(), NULL, 0);
    if (wlen <= 0) return s;
    std::wstring w(wlen, 0);
    MultiByteToWideChar(932, 0, s.data(), (int)s.size(), &w[0], wlen);
    int ulen = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), NULL, 0, NULL, NULL);
    if (ulen <= 0) return s;
    std::string u(ulen, 0);
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), &u[0], ulen, NULL, NULL);
    return u;
}
#endif

static bool load_csv(const std::string& path, CsvData& out) {
    bool ok = false;
    std::string content = read_whole_file(path, ok);
    if (!ok) return false;

    // UTF-8 BOM除去(3バイト連続一致時のみ、1回だけ)。半角カナ・全角文字のUTF-8
    // 1バイト目がたまたま0xEFと一致するケースを誤除去しないよう、3バイト全体で判定する。
    if (content.size() >= 3 && (unsigned char)content[0] == 0xEF &&
        (unsigned char)content[1] == 0xBB && (unsigned char)content[2] == 0xBF) {
        content = content.substr(3);
    }

    // 高-3: ヘッダ行のUTF-8妥当性を検査し、不正ならcp932フォールバックを試みる
    // (Windowsビルドのみ。非Windowス/CIビルドでは検査のみ行いフォールバックは省略)。
    {
        size_t nl = content.find('\n');
        std::string header_line = (nl == std::string::npos) ? content : content.substr(0, nl);
        if (!is_valid_utf8(header_line.data(), header_line.size())) {
#ifdef _WIN32
            content = cp932_to_utf8(content);
#else
            std::fprintf(stderr, "WARNING: CSVヘッダがUTF-8として不正です(cp932の可能性。"
                                  "このビルドではcp932自動変換に非対応です)\n");
#endif
        }
    }

    auto rows = tokenize_csv(content);
    if (rows.empty()) return false;

    out.headers.clear();
    out.headers.reserve(rows[0].size());
    for (auto& h : rows[0]) out.headers.push_back(trim_ws(h));

    out.rows.clear();
    out.rows.reserve(rows.size() > 0 ? rows.size() - 1 : 0);
    for (size_t i = 1; i < rows.size(); i++) {
        if (rows[i].size() == 1 && rows[i][0].empty()) continue;  // 空行はスキップ
        out.rows.push_back(std::move(rows[i]));
    }
    return true;
}

// 高-10/低-1: std::stod の部分パース(例:"12abc"→12として使ってしまう)を避けるため、
// フィールド全体が数値として消費されたかを確認する。加えて、パース結果が非有限
// (inf/-inf、"Infinity"等の文字列表現を含む)であれば NaN 扱いにする(→ median補完へ)。
// これにより C++ / predict_template.py(inf→NaN置換) / JS(Number.isFinite) の
// 3実装で非有限値の扱いを統一する(低-1)。
static double parse_numeric_field(const std::string& raw) {
    std::string t = trim_ws(raw);
    if (t.empty()) return std::numeric_limits<double>::quiet_NaN();
    try {
        size_t consumed = 0;
        double v = std::stod(t, &consumed);
        while (consumed < t.size() && std::isspace((unsigned char)t[consumed])) consumed++;
        if (consumed != t.size()) return std::numeric_limits<double>::quiet_NaN();
        if (!std::isfinite(v)) return std::numeric_limits<double>::quiet_NaN();
        return v;
    } catch (...) {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

// CSVフィールドの引用符要否判定+エスケープ(web/predict_template.html の csvField
// と同一仕様: カンマ・ダブルクォート・改行を含む場合のみクォートし、内部の"は""に)。
static std::string csv_field(const std::string& v) {
    if (v.find_first_of(",\"\n\r") == std::string::npos) return v;
    std::string out = "\"";
    for (char c : v) { if (c == '"') out += "\"\""; else out += c; }
    out += "\"";
    return out;
}

// ── main ─────────────────────────────────────────────────────────────────────
// 実体は run() に置き、main()/wmain() はそれを try/catch で包むだけにする(低-M12)。
// 破損した.treg(境界検査をすり抜けた巨大なn_fc_clip等)やLGBM循環参照検知が
// std::bad_alloc/length_error/std::runtime_error を送出すると、以前は未捕捉のまま
// abort()していた(exit=134、-mwindowsビルドではメッセージも出ず無言クラッシュ)。
// ここで捕捉しfatal()でユーザーに知らせる。
static int run(int argc, char* argv[]) {
    if (argc < 2) {
        fatal("CSVファイルをこのEXEにドラッグ＆ドロップしてください。");
        return 1;
    }
    const char* csv_path = argv[1];

    // exe自身のパス: Windowsでは GetModuleFileNameW で確実に取得する(argv[0]は
    // 呼び出し側依存で不完全なことがある、中-M2)。それ以外のOSでは従来通り argv[0]。
    std::string exe_path;
#ifdef _WIN32
    exe_path = get_exe_path_utf8();
    if (exe_path.empty()) exe_path = argv[0];
#else
    exe_path = argv[0];
#endif

    // .treg を読み込む: 1) 自己埋め込み  2) argv[2]  3) exe 隣の trained_model/model.treg
    TregModel model;
    bool loaded = false;

    auto embedded = load_embedded_treg(exe_path);
    if (!embedded.empty()) {
        loaded = load_treg(embedded.data(), embedded.size(), model);
    }

    if (!loaded) {
        std::string treg_path;
        if (argc >= 3) {
            treg_path = argv[2];
        } else {
            auto sep = exe_path.find_last_of("/\\");
            std::string dir = (sep != std::string::npos) ? exe_path.substr(0, sep + 1) : "./";
            treg_path = dir + "trained_model/model.treg";
        }
        bool ok = false;
        std::string buf = read_whole_file(treg_path, ok);
        if (!ok) {
            std::string msg = "モデルファイルが見つかりません:\n" + treg_path;
            fatal(msg.c_str());
            return 1;
        }
        loaded = load_treg(reinterpret_cast<const uint8_t*>(buf.data()), buf.size(), model);
    }

    if (!loaded) {
        fatal("モデルの読み込みに失敗しました。");
        return 1;
    }

    // CSV 解析(引用符・埋め込みカンマ・引用符内改行に対応。高-2)
    CsvData csv;
    if (!load_csv(csv_path, csv) || csv.headers.empty()) {
        fatal("CSVファイルを読み込めませんでした。\nファイルが壊れていないか確認してください。");
        return 1;
    }

    // 高-3: 学習時の特徴量列がCSVに欠けていないか検査する(blendは全メンバーのunion)。
    // 一部欠損は stderr 警告 + 出力CSV先頭のコメント行で明示して続行(median補完)、
    // 必要列が全て欠損している場合は fatal で終了する(補完しても無意味なため)。
    std::unordered_map<std::string, size_t> header_idx;
    for (size_t i = 0; i < csv.headers.size(); i++) header_idx[csv.headers[i]] = i;

    auto required = required_raw_columns_unique(model);
    std::vector<std::string> missing;
    for (auto& c : required) if (!header_idx.count(c)) missing.push_back(c);

    std::string warning_comment_line;
    if (!missing.empty()) {
        if (!required.empty() && missing.size() == required.size()) {
            std::string msg = "予測に必要な列が全てCSVに見つかりません:\n";
            for (auto& c : missing) msg += "  " + c + "\n";
            fatal(msg.c_str());
            return 1;
        }
        std::string joined;
        for (size_t i = 0; i < missing.size(); i++) { if (i) joined += ", "; joined += missing[i]; }
        std::fprintf(stderr, "WARNING: 学習時の列がCSVにありません(中央値で補完して続行します): %s\n",
                     joined.c_str());
        warning_comment_line = "# WARNING: missing columns (median-imputed): " + joined;
    }

    // 行データを CSV ヘッダ列順の std::vector<double> として1回だけ数値化する
    // (低-4: 以前の行ごと std::map<std::string,double> 構築をやめ、列名→インデックス
    // 解決を header_idx に一本化することで、大きなCSVでのメモリ・速度を改善)。
    std::vector<std::vector<double>> row_vals_all(csv.rows.size());
    for (size_t ri = 0; ri < csv.rows.size(); ri++) {
        const auto& raw = csv.rows[ri];
        auto& rv = row_vals_all[ri];
        rv.assign(csv.headers.size(), std::numeric_limits<double>::quiet_NaN());
        size_t m = std::min(raw.size(), csv.headers.size());
        for (size_t c = 0; c < m; c++) rv[c] = parse_numeric_field(raw[c]);
    }

    // 欠損補完して予測(Y逆変換・smearing補正・観測レンジclip・整数丸めは predict() 内で
    // 適用済み。blendの場合はメンバーごとの逆変換 → 外側の後処理、の順で再帰的に行われる)。
    std::vector<float> preds(csv.rows.size());
    for (size_t ri = 0; ri < csv.rows.size(); ri++)
        preds[ri] = predict(model, row_vals_all[ri], header_idx);

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

    // 中-M6: ヘッダに target_col が既に存在する場合は追記でなく該当フィールドを
    // 置換する(HTML/Python版 df[target_col]=... と同じ挙動に統一)。
    int target_idx = -1;
    for (size_t i = 0; i < csv.headers.size(); i++) {
        if (csv.headers[i] == model.target_col) { target_idx = (int)i; break; }
    }
    std::vector<std::string> out_headers = csv.headers;
    if (target_idx < 0) out_headers.push_back(model.target_col);

    // 高-2: 出力もトークナイズ済みのフィールド配列から再構成する(quote-awareな
    // csv_field()で必要な場合のみ引用符化)ため、引用符内改行を含む入力行があっても
    // 予測値との行対応がずれない(以前の「入力を生テキストのまま行単位で再読込して
    // 予測配列とインデックスで対応付ける」実装は、引用符内改行で行番号がずれた)。
    std::string content_out;
    if (!warning_comment_line.empty()) content_out += warning_comment_line + "\r\n";
    {
        std::string line;
        for (size_t i = 0; i < out_headers.size(); i++) {
            if (i) line += ",";
            line += csv_field(out_headers[i]);
        }
        content_out += line + "\r\n";
    }
    char numbuf[64];
    for (size_t ri = 0; ri < csv.rows.size(); ri++) {
        std::vector<std::string> out_row = csv.rows[ri];
        std::snprintf(numbuf, sizeof(numbuf), "%.9g", (double)preds[ri]);
        if (target_idx < 0) {
            out_row.push_back(numbuf);
        } else {
            if ((size_t)target_idx >= out_row.size()) out_row.resize(target_idx + 1);
            out_row[target_idx] = numbuf;
        }
        std::string line;
        for (size_t i = 0; i < out_row.size(); i++) {
            if (i) line += ",";
            line += csv_field(out_row[i]);
        }
        content_out += line + "\r\n";
    }

    FILE* out_f = fopen_utf8(out_path, "wb");
    if (!out_f) {
        std::string msg = "結果ファイルを作成できませんでした:\n" + out_path;
        fatal(msg.c_str());
        return 1;
    }
    bool write_ok = std::fwrite(content_out.data(), 1, content_out.size(), out_f) == content_out.size();
    std::fclose(out_f);
    if (!write_ok) {
        std::string msg = "結果ファイルの書き込みに失敗しました:\n" + out_path;
        fatal(msg.c_str());
        return 1;
    }

    // 完了通知は表示しない。結果はカレントフォルダ（CSVと同じ場所）へ静かに出力する。
    return 0;
}

#ifdef _WIN32
// 中-M2: cp932外の文字を含むCSVパスのドラッグ＆ドロップに対応する。main の wmain 化は
// この環境のMinGW(mingw.org系、mingw-w64ではない)では専用のCRTスタートアップ
// (wmainCRTStartup)を持たず、-municode オプションも存在しないため実現できなかった。
// 代わりに通常の main() のまま、OSが保持している「本当の」コマンドラインを
// GetCommandLineW() + CommandLineToArgvW() で取得しUTF-8へ変換する(タスクの代替案)。
// narrow の argv はOSの現在のANSIコードページ経由で欠落しうるため使わない
// (取得に失敗した場合のみ narrow argv にフォールバックする)。
static std::vector<std::string> get_argv_utf8_win32(int fallback_argc, char* fallback_argv[]) {
    std::vector<std::string> out;
    int wargc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (wargv) {
        out.reserve(wargc);
        for (int i = 0; i < wargc; i++) out.push_back(wide_to_utf8(wargv[i], -1));
        LocalFree(wargv);
    } else {
        for (int i = 0; i < fallback_argc; i++) out.push_back(fallback_argv[i] ? fallback_argv[i] : "");
    }
    return out;
}
#endif

int main(int argc, char* argv[]) {
    try {
#ifdef _WIN32
        std::vector<std::string> args_storage = get_argv_utf8_win32(argc, argv);
        std::vector<char*> args_c(args_storage.size());
        for (size_t i = 0; i < args_storage.size(); i++)
            args_c[i] = const_cast<char*>(args_storage[i].c_str());
        return run((int)args_c.size(), args_c.data());
#else
        return run(argc, argv);
#endif
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
