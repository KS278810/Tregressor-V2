"""_ag_benchmark/diagnose_losses.py

AGに大差で負けた例(real_cpu +0.183 / real_winequality +0.089 / count_poisson +0.035)の
真因を切り分ける。同一 train/test 分割で:
  (A) 標準的な単体モデル(Ridge / RandomForest / LightGBM / XGBoost / CatBoost)を学習し test R²
  (B) AG を再学習し「モデル別 test R² leaderboard」を取得(どの族が効いているか)
を比較する。treg の中核は LightGBM(+Ridge/GP/MLP blend)なので、
  ・CatBoost/XGB 単体 >> LightGBM 単体 なら → 「treg のモデル族の欠落」が真因
  ・単体GBMは横並びで AG stacked だけ勝つ なら → 「スタッキング深さ」が真因
  ・LightGBM を強化すると届く なら → 「treg の LGBM 設定が非力」が真因
AG venv で実行。
"""
import warnings
warnings.filterwarnings("ignore")
import os, sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import bench_common as bc

STANDALONE = ["real_cpu", "real_winequality", "count_poisson"]
AG_LEADERBOARD = ["real_cpu", "real_winequality"]
VAL_FRAC = 0.2
SEED = 0


def load_xy(name):
    wd = os.path.join(bc.WORK_DIR, name)
    tr = bc.read_any_csv(os.path.join(wd, "train.csv"))
    te = bc.read_any_csv(os.path.join(wd, "test.csv"))
    # manifest から target を引く
    man = bc.load_manifest("manifest_synth.json", "manifest_real.json")
    target = man[name]["target"]
    Xtr_raw, ytr = tr.drop(columns=[target]), tr[target].values
    Xte_raw, yte = te.drop(columns=[target]), te[target].values
    # object 列を one-hot(train/test 整合)
    comb = pd.concat([Xtr_raw.assign(_s=0), Xte_raw.assign(_s=1)], ignore_index=True)
    obj = [c for c in comb.columns if comb[c].dtype == object and c != "_s"]
    comb = pd.get_dummies(comb, columns=obj, dummy_na=True) if obj else comb
    comb = comb.fillna(comb.median(numeric_only=True))
    Xtr = comb[comb["_s"] == 0].drop(columns=["_s"]).values
    Xte = comb[comb["_s"] == 1].drop(columns=["_s"]).values
    return Xtr, ytr, Xte, yte, target


def split_val(Xtr, ytr):
    n = len(Xtr)
    perm = np.random.RandomState(SEED).permutation(n)
    k = max(20, int(n * VAL_FRAC))
    vi, ti = perm[:k], perm[k:]
    return Xtr[ti], ytr[ti], Xtr[vi], ytr[vi]


def r2(y, p):
    return round(bc.score(y, p)[0], 4)


def standalone(name):
    Xtr, ytr, Xte, yte, _ = load_xy(name)
    xt, yt, xv, yv = split_val(Xtr, ytr)
    out = {}

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr, ytr)
    out["Ridge"] = r2(yte, m.predict(Xte))

    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
    out["RandomForest"] = r2(yte, m.predict(Xte))

    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                          subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1)
    m.fit(xt, yt, eval_set=[(xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
    out["LightGBM"] = r2(yte, m.predict(Xte))

    import xgboost as xgb
    m = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                         early_stopping_rounds=50, eval_metric="rmse")
    m.fit(xt, yt, eval_set=[(xv, yv)], verbose=False)
    out["XGBoost"] = r2(yte, m.predict(Xte))

    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=6, random_seed=SEED,
                          verbose=False)
    m.fit(xt, yt, eval_set=(xv, yv), early_stopping_rounds=50)
    out["CatBoost"] = r2(yte, m.predict(Xte))

    return out, len(ytr), len(yte)


def ag_leaderboard(name, budget):
    from autogluon.tabular import TabularPredictor
    import shutil
    wd = os.path.join(bc.WORK_DIR, name)
    man = bc.load_manifest("manifest_synth.json", "manifest_real.json")
    target = man[name]["target"]
    tr = bc.read_any_csv(os.path.join(wd, "train.csv"))
    te = bc.read_any_csv(os.path.join(wd, "test.csv"))
    p = os.path.join(wd, "ag_diag")
    shutil.rmtree(p, ignore_errors=True)
    pred = TabularPredictor(label=target, problem_type="regression", eval_metric="r2",
                            path=p, verbosity=0).fit(
        tr, time_limit=budget, presets="good_quality",
        dynamic_stacking=False, ag_args_ensemble={"fold_fitting_strategy": "sequential_local"})
    lb = pred.leaderboard(te, silent=True)[["model", "score_test", "score_val"]]
    shutil.rmtree(p, ignore_errors=True)
    return lb


def treg_thorough(name):
    for r_ in bc.read_jsonl(os.path.join(bc.RESULTS_DIR, "treg_records.jsonl")):
        if r_["dataset"] == name:
            th = r_["modes"]["thorough"]
            return th["test_r2"], th["best_model"], th["train_sec"]
    return None, None, None


def main():
    budgets = {r["dataset"]: r.get("time_budget_sec")
               for r in bc.read_jsonl(os.path.join(bc.RESULTS_DIR, "ag_records.jsonl"))}
    print("=" * 72)
    print("(A) 単体モデル test R²(treg中核=LightGBM。同一train/test分割)")
    print("=" * 72)
    for name in STANDALONE:
        out, ntr, nte = standalone(name)
        tt, tmodel, _ = treg_thorough(name)
        print(f"\n■ {name}  (train={ntr} / test={nte})")
        print(f"   treg thorough = {tt}  (選択: {tmodel})")
        for k, v in sorted(out.items(), key=lambda kv: -(kv[1] if kv[1] is not None else -9)):
            mark = "  ← treg中核" if k == "LightGBM" else ""
            print(f"   {k:14s} = {v}{mark}")

    print("\n" + "=" * 72)
    print("(B) AutoGluon モデル別 test R² leaderboard(どの族が効いているか)")
    print("=" * 72)
    for name in AG_LEADERBOARD:
        b = budgets.get(name, 120)
        lb = ag_leaderboard(name, b)
        tt, _, _ = treg_thorough(name)
        print(f"\n■ {name}  (AG budget={b}s, treg thorough={tt})")
        for _, row in lb.head(12).iterrows():
            print(f"   {row['model']:26s} test={row['score_test']:.4f}  val={row['score_val']:.4f}")


if __name__ == "__main__":
    main()
