"""_ag_benchmark/run_ag.py

AutoGluon TabularPredictor を、T-regressor と「同一の train/test 分割」「同一の時間バジェット
(= 各問題の treg thorough 実測学習時間)」で全問学習し、共通 test を予測して中立採点する。
AG venv で実行(treg 実行が終わり results/treg_records.jsonl が揃ってから):
    agenv/Scripts/python.exe _ag_benchmark/run_ag.py

公平性:
  - 分割は treg が results/_work/<name>/ に書いた train.csv / test.csv / test_features.csv を
    そのまま読む(再分割しない = 同一分割を厳密に保証)
  - 採点は bench_common.score(treg と同一コード)
  - 時間バジェット T = max(FLOOR, round(treg thorough train_sec))。preset は good_quality
    (bagging+stacking を time_limit の許す範囲で行う = 同時間で本気のAG)

結果は results/ag_records.jsonl に1問1行で追記(再開可能)。
"""
import os
import shutil
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import bench_common as bc

TIME_FLOOR = 60          # AGに与える最小秒数(good_qualityが動く余地を残す)
PRESET = "good_quality"
AG_RECORDS = os.path.join(bc.RESULTS_DIR, "ag_records.jsonl")

# Windows安定化: Ray並列(raylet)がこの環境でクラッシュするため fold は逐次ローカルで学習。
# dynamic_stacking はスタッキング検出の予備fitに予算を食われ小バジェットで破綻するため無効化。
# いずれも「同一 time_limit 内で AG が学習できるモデル数」を左右するだけで、公平性(同時間)は不変。
FIT_KW = dict(
    dynamic_stacking=False,
    ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
)


def treg_thorough_seconds():
    """dataset -> treg thorough の学習秒数(時間バジェットの元)。"""
    out = {}
    for r in bc.read_jsonl(os.path.join(bc.RESULTS_DIR, "treg_records.jsonl")):
        th = (r.get("modes") or {}).get("thorough") or {}
        if th.get("train_sec"):
            out[r["dataset"]] = th["train_sec"]
    return out


def run_one(name, meta, budget):
    from autogluon.tabular import TabularPredictor

    wd = os.path.join(bc.WORK_DIR, name)
    train_csv = os.path.join(wd, "train.csv")
    testf_csv = os.path.join(wd, "test_features.csv")
    test_csv = os.path.join(wd, "test.csv")
    if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
        print(f"  [AG] {name}: 分割ファイルなし(treg未実行) → スキップ")
        return None

    target = meta["target"]
    train_df = bc.read_any_csv(train_csv)
    testf_df = bc.read_any_csv(testf_csv)
    y_test = bc.read_any_csv(test_csv)[target].values

    ag_path = os.path.join(wd, "ag_model")
    shutil.rmtree(ag_path, ignore_errors=True)

    t0 = time.time()
    predictor = TabularPredictor(
        label=target, problem_type="regression", eval_metric="r2",
        path=ag_path, verbosity=0,
    ).fit(train_df, time_limit=budget, presets=PRESET, **FIT_KW)
    fit_sec = time.time() - t0

    preds = predictor.predict(testf_df).values
    r2, rmse, mae, n_used = bc.score(y_test, preds)

    # リーダーボード先頭(検証スコア最良)モデル名と学習モデル数
    try:
        lb = predictor.leaderboard(silent=True)
        best_model = str(lb.iloc[0]["model"])
        n_models = int(len(lb))
        val_score = float(lb.iloc[0]["score_val"])
    except Exception:
        best_model, n_models, val_score = None, None, None

    rec = {
        "dataset": name, "target": target, "source": meta.get("source"),
        "family": meta.get("family"), "ceiling_r2": meta.get("ceiling_r2"),
        "test_r2": r2, "test_rmse": rmse, "test_mae": mae, "n_scored": n_used,
        "val_score_r2": round(val_score, 4) if val_score is not None else None,
        "best_model": best_model, "n_models": n_models,
        "time_budget_sec": budget, "fit_sec": round(fit_sec, 1),
        "preset": PRESET,
    }
    print(f"  [AG] {name}: test_r2={r2}  best={best_model}  models={n_models}  "
          f"budget={budget}s fit={fit_sec:.1f}s")
    shutil.rmtree(ag_path, ignore_errors=True)  # モデル成果物は嵩むので破棄
    return rec


def main():
    manifest = bc.load_manifest("manifest_synth.json", "manifest_real.json", "manifest_public.json")
    budgets = treg_thorough_seconds()
    done = {r["dataset"] for r in bc.read_jsonl(AG_RECORDS)}
    todo = [(n, m) for n, m in manifest.items() if n not in done]
    print(f"[AG] 対象 {len(todo)} 問(スキップ済 {len(done)}) preset={PRESET} floor={TIME_FLOOR}s")
    for i, (name, meta) in enumerate(todo, 1):
        budget = max(TIME_FLOOR, round(budgets.get(name, TIME_FLOOR)))
        print(f"[AG] ({i}/{len(todo)}) {name}  budget={budget}s")
        try:
            rec = run_one(name, meta, budget)
        except Exception as e:
            rec = {"dataset": name, "error": str(e)[:600], "time_budget_sec": budget}
            print(f"  [AG] {name}: ERROR {str(e)[:200]}")
        if rec is not None:
            bc.append_jsonl(AG_RECORDS, rec)
    print(f"[AG] 完了。結果 -> {AG_RECORDS}")


if __name__ == "__main__":
    main()
