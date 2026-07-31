"""_ag_benchmark/diagnose_losses2.py

40問版でAGに負けた問題(特にUCI公開10問中の6敗)の真因切り分け。仮説を実験で確定する:

(A) 単体モデル比較(Ridge/RF/LGBM/XGB/CatBoost、同一分割)
    → 「treg中核LGBMの素の実力」と「モデル族ギャップ」を分離
    対象: pub_parkinsons / pub_appliances / pub_realestate / pub_studentperf / pub_concrete

(B) appliances の date 列仮説の確定実験(3条件):
    B1: date列を除外してAG再学習(同予算) → AGの優位が消えれば「datetime解析の差」が真因
    B2: dateから hour/dayofweek/month を手動抽出して単体LGBM → 時刻信号の寄与上限を実測
    B3: date除外の単体LGBM(ベースライン)

(C) AG leaderboard(モデル別test R²): pub_parkinsons / pub_appliances
    → AGの勝ちを駆動しているモデル族を特定

AG venv で実行。結果は stdout(diagnose2.log)に。
"""
import warnings
warnings.filterwarnings("ignore")
import os, sys, shutil, time
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import bench_common as bc

SEED = 0
VAL_FRAC = 0.2


def load_xy(name, drop_cols=(), extra_fe=None):
    wd = os.path.join(bc.WORK_DIR, name)
    tr = bc.read_any_csv(os.path.join(wd, "train.csv"))
    te = bc.read_any_csv(os.path.join(wd, "test.csv"))
    man = bc.load_manifest("manifest_synth.json", "manifest_real.json", "manifest_public.json")
    target = man[name]["target"]
    for df in (tr, te):
        for c in drop_cols:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
    if extra_fe is not None:
        tr, te = extra_fe(tr), extra_fe(te)
    ytr, yte = tr[target].values, te[target].values
    Xtr_raw, Xte_raw = tr.drop(columns=[target]), te.drop(columns=[target])
    comb = pd.concat([Xtr_raw.assign(_s=0), Xte_raw.assign(_s=1)], ignore_index=True)
    obj = [c for c in comb.columns if comb[c].dtype == object and c != "_s"]
    if obj:
        comb = pd.get_dummies(comb, columns=obj, dummy_na=True)
    comb = comb.fillna(comb.median(numeric_only=True))
    Xtr = comb[comb["_s"] == 0].drop(columns=["_s"]).values.astype(float)
    Xte = comb[comb["_s"] == 1].drop(columns=["_s"]).values.astype(float)
    return Xtr, ytr, Xte, yte, target


def r2(y, p):
    return round(bc.score(y, p)[0], 4)


def split_val(Xtr, ytr):
    n = len(Xtr)
    perm = np.random.RandomState(SEED).permutation(n)
    k = max(20, int(n * VAL_FRAC))
    vi, ti = perm[:k], perm[k:]
    return Xtr[ti], ytr[ti], Xtr[vi], ytr[vi]


def standalone(name, drop_cols=(), extra_fe=None, models=("Ridge", "RF", "LGBM", "XGB", "CAT")):
    Xtr, ytr, Xte, yte, _ = load_xy(name, drop_cols, extra_fe)
    xt, yt, xv, yv = split_val(Xtr, ytr)
    out = {}
    if "Ridge" in models:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        out["Ridge"] = r2(yte, make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr, ytr).predict(Xte))
    if "RF" in models:
        from sklearn.ensemble import RandomForestRegressor
        out["RandomForest"] = r2(yte, RandomForestRegressor(n_estimators=400, n_jobs=-1,
                                                            random_state=SEED).fit(Xtr, ytr).predict(Xte))
    if "LGBM" in models:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                              subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1)
        m.fit(xt, yt, eval_set=[(xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
        out["LightGBM"] = r2(yte, m.predict(Xte))
    if "XGB" in models:
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                             early_stopping_rounds=50, eval_metric="rmse")
        m.fit(xt, yt, eval_set=[(xv, yv)], verbose=False)
        out["XGBoost"] = r2(yte, m.predict(Xte))
    if "CAT" in models:
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=6, random_seed=SEED, verbose=False)
        m.fit(xt, yt, eval_set=(xv, yv), early_stopping_rounds=50)
        out["CatBoost"] = r2(yte, m.predict(Xte))
    return out


def ag_run(name, budget, drop_cols=(), tag="diag"):
    from autogluon.tabular import TabularPredictor
    wd = os.path.join(bc.WORK_DIR, name)
    man = bc.load_manifest("manifest_synth.json", "manifest_real.json", "manifest_public.json")
    target = man[name]["target"]
    tr = bc.read_any_csv(os.path.join(wd, "train.csv"))
    te = bc.read_any_csv(os.path.join(wd, "test.csv"))
    for df in (tr, te):
        for c in drop_cols:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
    p = os.path.join(wd, f"ag_{tag}")
    shutil.rmtree(p, ignore_errors=True)
    pred = TabularPredictor(label=target, problem_type="regression", eval_metric="r2",
                            path=p, verbosity=0).fit(
        tr, time_limit=budget, presets="good_quality",
        dynamic_stacking=False, ag_args_ensemble={"fold_fitting_strategy": "sequential_local"})
    lb = pred.leaderboard(te, silent=True)[["model", "score_test", "score_val"]]
    best_test = r2(te[target].values, pred.predict(te.drop(columns=[target])).values)
    shutil.rmtree(p, ignore_errors=True)
    return best_test, lb


def appliances_datetime_fe(df):
    """dateから hour/dayofweek/month を抽出(元のdate列は除去)。"""
    df = df.copy()
    if "date" in df.columns:
        # '2016-01-1313:50:00' 形式(日付と時刻の間に空白なし)にも対応
        dt = pd.to_datetime(df["date"], format="%Y-%m-%d%H:%M:%S", errors="coerce")
        if dt.isna().mean() > 0.5:
            dt = pd.to_datetime(df["date"], errors="coerce")
        df["fe_hour"] = dt.dt.hour
        df["fe_dow"] = dt.dt.dayofweek
        df["fe_month"] = dt.dt.month
        df["fe_minutes"] = dt.dt.hour * 60 + dt.dt.minute
        df = df.drop(columns=["date"])
    return df


def treg_result(name):
    for r_ in bc.read_jsonl(os.path.join(bc.RESULTS_DIR, "treg_records.jsonl")):
        if r_["dataset"] == name:
            q, t = r_["modes"]["quick"], r_["modes"]["thorough"]
            return q["test_r2"], t["test_r2"], t["best_model"]
    return None, None, None


def main():
    budgets = {r["dataset"]: r.get("time_budget_sec")
               for r in bc.read_jsonl(os.path.join(bc.RESULTS_DIR, "ag_records.jsonl"))}
    ag_scores = {r["dataset"]: r.get("test_r2")
                 for r in bc.read_jsonl(os.path.join(bc.RESULTS_DIR, "ag_records.jsonl"))}

    print("=" * 76)
    print("(A) 単体モデル test R²(同一分割。treg中核=LightGBM)")
    print("=" * 76)
    for name in ["pub_parkinsons", "pub_appliances", "pub_realestate", "pub_studentperf", "pub_concrete"]:
        out = standalone(name)
        tq, tt, tmodel = treg_result(name)
        print(f"\n■ {name}  treg: quick={tq} thorough={tt}({tmodel})  AG={ag_scores.get(name)}")
        for k, v in sorted(out.items(), key=lambda kv: -(kv[1] if kv[1] is not None else -9)):
            mark = "  ← treg中核" if k == "LightGBM" else ""
            print(f"   {k:14s} = {v}{mark}")

    print("\n" + "=" * 76)
    print("(B) pub_appliances date列仮説の確定実験")
    print("=" * 76)
    b3 = standalone("pub_appliances", drop_cols=("date",), models=("LGBM",))
    print(f"  B3 date除外 + 単体LGBM         = {b3['LightGBM']}   (treg相当のベースライン)")
    b2 = standalone("pub_appliances", extra_fe=appliances_datetime_fe, models=("LGBM",))
    print(f"  B2 hour/dow/month抽出 + LGBM   = {b2['LightGBM']}   (時刻信号の寄与を実測)")
    bud = budgets.get("pub_appliances", 135)
    b1_test, b1_lb = ag_run("pub_appliances", bud, drop_cols=("date",), tag="nodate")
    print(f"  B1 date除外 + AG(同予算{bud}s)  = {b1_test}   (AG本来値 {ag_scores.get('pub_appliances')} との差=date寄与)")
    print("  B1 leaderboard(上位6):")
    for _, row in b1_lb.head(6).iterrows():
        print(f"     {row['model']:26s} test={row['score_test']:.4f}")

    print("\n" + "=" * 76)
    print("(C) AG leaderboard(モデル別test R²)— 勝ちを駆動する族の特定")
    print("=" * 76)
    for name in ["pub_parkinsons", "pub_appliances"]:
        bud = budgets.get(name, 130)
        best_test, lb = ag_run(name, bud, tag="lb")
        tq, tt, tmodel = treg_result(name)
        print(f"\n■ {name}  (budget={bud}s, treg thorough={tt})")
        for _, row in lb.head(12).iterrows():
            sv = row['score_val']
            svs = f"{sv:.4f}" if pd.notna(sv) else "nan"
            print(f"   {row['model']:28s} test={row['score_test']:.4f}  val={svs}")


if __name__ == "__main__":
    main()
