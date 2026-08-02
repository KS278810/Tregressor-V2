"""_ag_benchmark/gen_synthetic.py

T-regressor vs AutoGluon 比較ベンチ用の合成回帰問題 24 問を決定論的に生成する。

各問題は「決定論的シグナル s = E[y|X]」と「既約ノイズ e」に分解でき、
    ceiling_r2 = 1 - Var(y - s) / Var(y)
を manifest に記録する。これは「どんな学習器でも原理的に超えられない test R² の上限」で、
各ツールの到達 R² を ceiling と比べることで“地力の絶対評価”ができる(pure_noise は 0)。

numpy/pandas のみに依存(T-regressor の python-embed でそのまま実行可能)。
出力: _ag_benchmark/data/<name>.csv + data/manifest_synth.json
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data")


def _rng(seed):
    return np.random.RandomState(seed)


def _ceiling(y, signal):
    """ceiling_r2 = 1 - Var(y - E[y|X]) / Var(y)。y と同型の deterministic signal を渡す。"""
    y = np.asarray(y, dtype=float)
    s = np.asarray(signal, dtype=float)
    vy = np.var(y)
    if vy <= 0:
        return 0.0
    return float(round(1.0 - np.var(y - s) / vy, 4))


# 返り値: (df, target_col, signal_array, family, note)
def linear_clean(n=800, seed=101):
    rng = _rng(seed)
    X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(1, 6)})
    s = 3.0 * X.x1 - 2.0 * X.x2 + 0.5 * X.x3 + 0.8 * X.x4 - 1.0 * X.x5
    y = s + rng.randn(n) * 0.2
    X["y"] = y
    return X, "y", s.values, "linear", "きれいな線形・低ノイズ"


def linear_noisy(n=800, seed=102):
    rng = _rng(seed)
    X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(1, 6)})
    s = 3.0 * X.x1 - 2.0 * X.x2 + 0.5 * X.x3 + 0.8 * X.x4 - 1.0 * X.x5
    y = s + rng.randn(n) * 3.0
    X["y"] = y
    return X, "y", s.values, "linear", "線形だが高ノイズ(ceiling低)。過信しないか"


def linear_highdim_sparse(n=600, p=60, seed=103):
    rng = _rng(seed)
    X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(1, p + 1)})
    coefs = np.zeros(p)
    relevant = [0, 5, 12, 23, 40, 55]
    coefs[relevant] = [3.0, -2.5, 2.0, -1.5, 1.8, -1.2]
    s = X.values @ coefs
    y = s + rng.randn(n) * 0.5
    X["y"] = y
    return X, "y", s, "linear", f"高次元({p}特徴)中6本のみ有効。特徴選抜"


def collinear(n=600, seed=104):
    rng = _rng(seed)
    z1, z2, z3 = rng.randn(n), rng.randn(n), rng.randn(n)
    cols = {}
    for i, z in enumerate([z1, z2, z3]):
        for j in range(4):  # 各潜在から4つの相関観測変数
            cols[f"x{i}_{j}"] = z + rng.randn(n) * 0.1
    X = pd.DataFrame(cols)
    s = 2.0 * z1 - 1.5 * z2 + 1.0 * z3
    y = s + rng.randn(n) * 0.4
    X["y"] = y
    return X, "y", s, "linear", "強い多重共線性(3潜在→12観測)"


def nonlinear_interaction(n=900, seed=105):
    rng = _rng(seed)
    X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(1, 7)})
    s = (np.sin(X.x1 * 2) * 3 + X.x2 * X.x3 + np.where(X.x4 > 0, 2.0, -2.0) + 0.4 * X.x5 ** 2)
    y = s + rng.randn(n) * 1.2
    X["y"] = y
    return X, "y", s.values, "nonlinear", "非線形+交互作用+段差"


def multiplicative(n=700, seed=106):
    rng = _rng(seed)
    X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(1, 5)})
    s = 2.0 * X.x1 * X.x2 + 1.5 * X.x3 * X.x4
    y = s + rng.randn(n) * 0.5
    X["y"] = y
    return X, "y", s.values, "nonlinear", "純交互作用(主効果ゼロ)。線形に厳しい"


def trig_smooth(n=700, seed=107):
    rng = _rng(seed)
    X = pd.DataFrame({f"x{i}": rng.uniform(-3, 3, n) for i in range(1, 4)})
    s = 2 * np.sin(X.x1) + 1.5 * np.cos(X.x2) + np.sin(X.x1 + X.x3)
    y = s + rng.randn(n) * 0.4
    X["y"] = y
    return X, "y", s.values, "nonlinear", "滑らかな三角関数曲面(GP向き)"


def piecewise_steps(n=700, seed=108):
    rng = _rng(seed)
    X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(1, 5)})
    s = (3.0 * (X.x1 > 0.5) - 2.0 * (X.x2 < -0.3) + 1.5 * ((X.x3 > 0) & (X.x4 > 0)).astype(float))
    y = s + rng.randn(n) * 0.4
    X["y"] = y
    return X, "y", s.values, "nonlinear", "階段/閾値関数(木に有利・線形に不利)"


def xor_sign(n=700, seed=109):
    rng = _rng(seed)
    X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(1, 5)})
    s = 2.0 * np.sign(X.x1) * np.sign(X.x2) + 1.5 * np.sign(X.x3) * np.sign(X.x4)
    y = s + rng.randn(n) * 0.5
    X["y"] = y
    return X, "y", s.values, "nonlinear", "XOR型符号交互作用(線形相関ほぼ0)"


def polynomial_deg3(n=700, seed=110):
    rng = _rng(seed)
    X = pd.DataFrame({f"x{i}": rng.uniform(-2, 2, n) for i in range(1, 4)})
    s = X.x1 ** 3 - 2 * X.x2 ** 2 + X.x1 * X.x2 + 0.5 * X.x3
    y = s + rng.randn(n) * 1.0
    X["y"] = y
    return X, "y", s.values, "nonlinear", "3次多項式(poly特徴が効く)"


def heteroscedastic(n=700, seed=111):
    rng = _rng(seed)
    X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(1, 4)})
    s = 2.0 * X.x1 - 1.0 * X.x2 + 0.5 * X.x3
    noise_sd = 0.3 + 0.8 * np.abs(X.x1)
    y = s + rng.randn(n) * noise_sd
    X["y"] = y
    return X, "y", s.values, "linear", "平均は線形だがノイズ分散がxに依存(不均一分散)"


def monotonic_saturating(n=700, seed=112):
    rng = _rng(seed)
    X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(1, 4)})
    s = 4 * np.tanh(1.5 * X.x1) + 2 / (1 + np.exp(-2 * X.x2)) + 0.5 * X.x3
    y = s + rng.randn(n) * 0.4
    X["y"] = y
    return X, "y", s.values, "nonlinear", "単調飽和(tanh/ロジスティック)"


def radial_rbf(n=600, seed=113):
    rng = _rng(seed)
    X = pd.DataFrame({f"x{i}": rng.uniform(-2.5, 2.5, n) for i in range(1, 4)})
    s = 5 * np.exp(-(X.x1 ** 2 + X.x2 ** 2) / 2.0) + 3 * np.exp(-((X.x3 - 1) ** 2) / 1.0)
    y = s + rng.randn(n) * 0.35
    X["y"] = y
    return X, "y", s.values, "nonlinear", "動径基底(距離の関数、GP向き)"


def many_irrelevant(n=600, p=45, seed=114):
    rng = _rng(seed)
    X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(1, p + 1)})
    s = 3.0 * X.x1 - 2.0 * X.x2 + 1.5 * np.sin(X.x3 * 2) + 1.0 * (X.x4 > 0)
    y = s + rng.randn(n) * 0.5
    X["y"] = y
    return X, "y", s.values, "nonlinear", f"有効4本+無関係{p-4}本のダミー特徴"


def categorical_low(n=600, seed=115):
    rng = _rng(seed)
    classes = list("ABCDE")
    effects = _rng(seed + 1).randn(len(classes)) * 4
    idx = rng.randint(0, len(classes), size=n)
    grade = np.array(classes)[idx]
    x1 = rng.randn(n)
    s = 1.5 * x1 + effects[idx]
    y = s + rng.randn(n) * 0.5
    return pd.DataFrame({"x1": x1, "grade": grade, "y": y}), "y", s, "categorical", "低カード(5)one-hot経路"


def categorical_high(n=600, seed=116):
    rng = _rng(seed)
    n_city = 30
    cities = [f"city_{i}" for i in range(n_city)]
    effects = _rng(seed + 1).randn(n_city) * 4
    idx = rng.randint(0, n_city, size=n)
    city = np.array(cities)[idx]
    x1 = rng.randn(n)
    raw_id = np.array([f"id_{i}" for i in range(n)])
    s = x1 + effects[idx]
    y = s + rng.randn(n) * 0.5
    return pd.DataFrame({"x1": x1, "city": city, "raw_id": raw_id, "y": y}), "y", s, "categorical", \
        "高カード(30)target-enc + id列(除外)"


def categorical_interaction(n=600, seed=117):
    rng = _rng(seed)
    groups = list("PQRS")
    slopes = {"P": 3.0, "Q": -2.0, "R": 0.5, "S": 1.5}
    idx = rng.randint(0, len(groups), size=n)
    grp = np.array(groups)[idx]
    x1 = rng.randn(n)
    slope_arr = np.array([slopes[g] for g in grp])
    s = slope_arr * x1  # 傾きがカテゴリごとに変わる
    y = s + rng.randn(n) * 0.5
    return pd.DataFrame({"x1": x1, "grp": grp, "y": y}), "y", s, "categorical", \
        "x1の傾きがカテゴリで変わる(cat×num交互作用)"


def mixed_messy(n=600, seed=118):
    rng = _rng(seed)
    flag = rng.rand(n) < 0.5
    cats = list("abcdef")
    ceff = _rng(seed + 2).randn(len(cats)) * 3
    cidx = rng.randint(0, len(cats), size=n)
    cat = np.array(cats)[cidx]
    x1, x2, x3 = rng.randn(n), rng.randn(n), rng.randn(n)
    s = 2 * x1 - x2 + 0.5 * x3 * x1 + np.where(flag, 3.0, -3.0) + ceff[cidx]
    y = s + rng.randn(n) * 0.5
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "flag": flag, "cat": cat, "y": y})
    for col in ["x2", "x3"]:  # 25% 欠損
        df.loc[rng.rand(n) < 0.25, col] = np.nan
    return df, "y", s, "mixed", "bool+cat+数値+欠損の現実的な混在"


def skewed_target(n=600, seed=119):
    rng = _rng(seed)
    x1, x2 = rng.randn(n), rng.randn(n)
    mu = 1.0 + 0.7 * x1 + 0.5 * x2
    sig = 0.3
    y = np.exp(mu + rng.randn(n) * sig)
    s = np.exp(mu + sig ** 2 / 2.0)  # E[y|x] for lognormal
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y}), "y", s, "pathology", "対数正規の歪んだy(y変換分岐)"


def outlier_contaminated(n=600, seed=120):
    rng = _rng(seed)
    x1, x2 = rng.randn(n), rng.randn(n)
    s = 2 * x1 - x2
    y = s + rng.randn(n) * 0.5
    n_out = max(3, int(n * 0.02))
    out_idx = rng.choice(n, size=n_out, replace=False)
    y = y.copy()
    y[out_idx] += rng.choice([1.0, -1.0], size=n_out) * rng.uniform(30, 50, size=n_out)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y}), "y", s, "pathology", "2%のyに巨大外れ値(頑健性)"


def count_poisson(n=700, seed=121):
    rng = _rng(seed)
    x1, x2 = rng.randn(n), rng.randn(n)
    lam = np.exp(1.0 + 0.5 * x1 - 0.3 * x2)
    y = rng.poisson(lam).astype(float)
    s = lam  # E[y|x] = lambda
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y}), "y", s, "pathology", "ポアソン計数(整数・歪み)"


def interaction_deep(n=800, seed=122):
    rng = _rng(seed)
    X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(1, 6)})
    s = X.x1 * X.x2 * X.x3 + 1.0 * X.x4 - 0.5 * X.x5
    y = s + rng.randn(n) * 0.5
    X["y"] = y
    return X, "y", s.values, "nonlinear", "3方向交互作用(加法モデルに極めて不利)"


def small_n(n=40, seed=123):
    rng = _rng(seed)
    X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(1, 5)})
    s = X.x1 - X.x2 + 0.5 * X.x3
    y = s + rng.randn(n) * 0.3
    X["y"] = y
    return X, "y", s.values, "pathology", "小標本(n=40、過学習リスク)"


def pure_noise(n=600, seed=124):
    rng_x = _rng(seed)
    rng_y = _rng(seed + 1000)
    X = pd.DataFrame({f"x{i}": rng_x.randn(n) for i in range(1, 9)})
    y = rng_y.randn(n)
    X["y"] = y
    return X, "y", np.zeros(n), "pathology", "リーク検知(Xとy完全独立、ceiling=0)"


GENERATORS = [
    linear_clean, linear_noisy, linear_highdim_sparse, collinear,
    nonlinear_interaction, multiplicative, trig_smooth, piecewise_steps,
    xor_sign, polynomial_deg3, heteroscedastic, monotonic_saturating,
    radial_rbf, many_irrelevant, categorical_low, categorical_high,
    categorical_interaction, mixed_messy, skewed_target, outlier_contaminated,
    count_poisson, interaction_deep, small_n, pure_noise,
]


# ── モデル選択の妥当性チェック用メタデータ ─────────────────────────────────
# 2026-08: 「結果(R²)の比較だけでなく、意図通りのモデルが選ばれているか」を自動判定
# したいという要望に対応。ただしT-regressorの"Linear"は内部でLGBM screening/derived
# features(交互作用項の自動生成)を伴うため、"family=nonlinearだからLinearが勝ったら
# 即エラー"のような単純な family⇔model の1対1判定は誤検知を生む(例: multiplicative
# はfamily=nonlinearだがLinear(Ridge)がceiling比98.5%で勝っており、これは
# derived interaction featuresが効いているためで異常ではない)。
# そのため、ここでは「生成関数のnote自体が特定モデル種別を明示的に予見している」
# 狭いケースだけに絞って expected_models(許容モデル種別の集合)を宣言する。
# 該当しないデータセットは None のままとし、report.md では家系(family)と選ばれた
# モデルを併記するに留め、判定は下さない(過剰判定による誤検知を避ける)。
#
# model_type の取りうる値: linear / linear_poly / lgbm / gp / mlp / rf / xt / blend
# (train_bridge.py の _TREG_TYPE_MAP 等を参照。lgbm_bag は lgbm 扱いとして照合する)
EXPECTED_MODELS = {
    # note: "動径基底(距離の関数、GP向き)" — RBFカーネルGPが理論的に最適
    "radial_rbf":            ["gp", "blend"],
    # note: "滑らかな三角関数曲面(GP向き)"
    "trig_smooth":            ["gp", "blend"],
    # note: "階段/閾値関数(木に有利・線形に不利)" — 決定木系が閾値を素直に表現できる
    "piecewise_steps":        ["lgbm", "rf", "xt", "blend"],
    # note: "XOR型符号交互作用(線形相関ほぼ0)" — 線形相関がほぼ0なので木/GP/MLP系が必要
    "xor_sign":               ["lgbm", "rf", "xt", "gp", "mlp", "blend"],
    # note: "強い多重共線性" — Ridgeの正則化が真価を発揮する典型ケース
    "collinear":              ["linear", "linear_poly", "blend"],
    # note: "高次元(60特徴)中6本のみ有効。特徴選抜" — スパース線形+スクリーニングの土俵
    "linear_highdim_sparse":  ["linear", "linear_poly", "blend"],
    # note: "リーク検知(Xとy完全独立、ceiling=0)" — モデル種別は問わず、
    # |test_r2| が小さく保たれているか(=リーク・過信をしていないか)だけを別途チェックする
    "pure_noise":             None,
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {}
    for fn in GENERATORS:
        name = fn.__name__
        df, target, signal, family, note = fn()
        csv_name = f"{name}.csv"
        df.to_csv(os.path.join(OUT_DIR, csv_name), index=False, encoding="utf-8")
        ceil = _ceiling(df[target].values, signal)
        manifest[name] = {
            "csv": csv_name, "target": target, "n": int(len(df)),
            "n_cols": int(df.shape[1] - 1), "family": family,
            "ceiling_r2": ceil, "source": "synthetic", "note": note,
            "expected_models": EXPECTED_MODELS.get(name),
        }
        print(f"生成: {name:24s} n={len(df):5d} cols={df.shape[1]-1:3d} ceiling_r2={ceil:.3f}  {note}")

    with open(os.path.join(OUT_DIR, "manifest_synth.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"\n完了: 合成 {len(manifest)} 問を {OUT_DIR} に生成。")


if __name__ == "__main__":
    main()
