"""diagnose_parkinsons_subject_encoding.py — 施策B-1(使い捨て実験、製品コード変更なし)

pub_parkinsonsの反復測定構造(31被験者、age+sexが実質的な被験者代理キー)に対して、
group-mean target encoding特徴を1本追加するだけでどれだけtest R²が改善するかを、
標準ライブラリ(agenv、製品のtrain_bridge.pyは使わない)で測定する。

実行: _ag_benchmark/agenv/Scripts/python.exe diagnose_parkinsons_subject_encoding.py
"""
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import lightgbm as lgb

TRAIN = "results/_work/pub_parkinsons/train.csv"
TEST = "results/_work/pub_parkinsons/test.csv"
TARGET = "target"
KEY_COLS = ["age", "sex"]

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)
feat_cols = [c for c in train.columns if c != TARGET]

# ── 被験者代理キーによるgroup-mean target encoding ──────────────────────────
global_mean = train[TARGET].mean()
group_mean = train.groupby(KEY_COLS)[TARGET].mean()


def add_subject_encoding(df):
    df = df.copy()
    key = list(zip(*(df[c] for c in KEY_COLS)))
    df["subject_mean_target"] = [group_mean.get(k, global_mean) for k in key]
    return df


train_enc = add_subject_encoding(train)
test_enc = add_subject_encoding(test)

y_train = train[TARGET].values
y_test = test[TARGET].values

print(f"train n={len(train)} test n={len(test)}")
n_groups = train.groupby(KEY_COLS).ngroups
print(f"(age,sex) 組み合わせ数(train): {n_groups}")
overlap = test.set_index(KEY_COLS).index.isin(train.set_index(KEY_COLS).index).mean()
print(f"testの(age,sex)がtrainに存在する割合: {overlap*100:.1f}%")

# ── 0. 単純ベースライン: 被験者平均だけ ──────────────────────────────────────
pred0 = test_enc["subject_mean_target"].values
r2_0 = r2_score(y_test, pred0)
print(f"\n[0] 被験者平均のみ: test R2={r2_0:.4f}")

# ── 1. LightGBM(特徴無し vs 被験者エンコーディング追加) ─────────────────────
def fit_predict_lgbm(Xtr, ytr, Xte):
    dtr = lgb.Dataset(Xtr, label=ytr)
    params = dict(objective="regression", num_leaves=31, learning_rate=0.03,
                  min_child_samples=10, verbose=-1, seed=42)
    bst = lgb.train(params, dtr, num_boost_round=500)
    return bst.predict(Xte)


pred1a = fit_predict_lgbm(train[feat_cols].values, y_train, test[feat_cols].values)
r2_1a = r2_score(y_test, pred1a)
print(f"\n[1a] LightGBM(元特徴のみ): test R2={r2_1a:.4f}")

feat_cols_enc = feat_cols + ["subject_mean_target"]
pred1b = fit_predict_lgbm(train_enc[feat_cols_enc].values, y_train, test_enc[feat_cols_enc].values)
r2_1b = r2_score(y_test, pred1b)
print(f"[1b] LightGBM(+被験者encoding): test R2={r2_1b:.4f}  (delta={r2_1b-r2_1a:+.4f})")

# ── 2. GP(元特徴 vs +被験者encoding、300行キャップは製品と合わせる) ─────────
GP_MAX_TRAIN = 300
rng = np.random.RandomState(42)


def fit_predict_gp(Xtr_df, ytr, Xte_df):
    if len(Xtr_df) > GP_MAX_TRAIN:
        idx = rng.choice(len(Xtr_df), GP_MAX_TRAIN, replace=False)
        Xtr_df = Xtr_df.iloc[idx]
        ytr = ytr[idx]
    scaler = StandardScaler().fit(Xtr_df.values)
    Xtr_s = scaler.transform(Xtr_df.values)
    Xte_s = scaler.transform(Xte_df.values)
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(Xtr_s.shape[1])) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
    gp.fit(Xtr_s, ytr)
    return gp.predict(Xte_s)


pred2a = fit_predict_gp(train[feat_cols], y_train, test[feat_cols])
r2_2a = r2_score(y_test, pred2a)
print(f"\n[2a] GP(元特徴のみ、{GP_MAX_TRAIN}行キャップ): test R2={r2_2a:.4f}")

pred2b = fit_predict_gp(train_enc[feat_cols_enc], y_train, test_enc[feat_cols_enc])
r2_2b = r2_score(y_test, pred2b)
print(f"[2b] GP(+被験者encoding、{GP_MAX_TRAIN}行キャップ): test R2={r2_2b:.4f}  (delta={r2_2b-r2_2a:+.4f})")

# ── 3. 単純平均ブレンド(LightGBM+被験者encoding版 と GP+被験者encoding版) ────
pred3 = 0.5 * pred1b + 0.5 * pred2b
r2_3 = r2_score(y_test, pred3)
print(f"\n[3] LightGBM(+enc) と GP(+enc) の単純平均ブレンド: test R2={r2_3:.4f}")

print("\n=== まとめ ===")
print(f"現行treg thorough(製品, 参考値): test R2=0.901")
print(f"AutoGluon(参考値): test R2=0.961")
print(f"[0] 被験者平均のみ:            {r2_0:.4f}")
print(f"[1a/1b] LightGBM無/有encoding: {r2_1a:.4f} -> {r2_1b:.4f} (delta {r2_1b-r2_1a:+.4f})")
print(f"[2a/2b] GP無/有encoding:       {r2_2a:.4f} -> {r2_2b:.4f} (delta {r2_2b-r2_2a:+.4f})")
print(f"[3] LGBM+GP(いずれもencoding付き)平均ブレンド: {r2_3:.4f}")
