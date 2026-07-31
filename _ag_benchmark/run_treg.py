"""_ag_benchmark/run_treg.py

T-regressor バックエンド(train_bridge.py)を quick/thorough 両モードで全問学習し、
共通ホールドアウト test を predict_template.py で予測して中立採点する。
embed python で実行:
    dist_portable/T-regressor/python-embed/python.exe _ag_benchmark/run_treg.py [manifest_synth.json ...]

結果は results/treg_records.jsonl に1問1行で追記(途中クラッシュしても部分結果が残る)。
既に treg_records.jsonl に存在する dataset はスキップ(再開可能)。
"""
import json
import os
import shutil
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import bench_common as bc

ROOT = bc.ROOT
EMBED_PY = os.path.join(ROOT, "dist_portable", "T-regressor", "python-embed", "python.exe")
TRAIN_BRIDGE = os.path.join(ROOT, "train_bridge.py")
PREDICT_TEMPLATE = os.path.join(ROOT, "predict_template.py")
LIGHT = os.path.join(ROOT, "_light.py")
TRAINED = os.path.join(ROOT, "trained_model")
BACKUP = os.path.join(bc.HERE, "_orig_trained_model")
TREG_RECORDS = os.path.join(bc.RESULTS_DIR, "treg_records.jsonl")
TRAIN_TIMEOUT = 1200


def backup_trained_model():
    if os.path.isdir(TRAINED) and not os.path.isdir(BACKUP):
        shutil.copytree(TRAINED, BACKUP)
        print(f"[treg] 元の trained_model を退避: {BACKUP}")


def treg_train(train_csv, target, strategy):
    t0 = time.time()
    p = subprocess.run(
        [EMBED_PY, TRAIN_BRIDGE, train_csv, target, "0", strategy, "4"],
        capture_output=True, cwd=ROOT, timeout=TRAIN_TIMEOUT, encoding="utf-8", errors="replace")
    elapsed = time.time() - t0
    rj = None
    for line in (p.stdout or "").splitlines():
        if line.startswith("RESULT_JSON:"):
            rj = json.loads(line[len("RESULT_JSON:"):])
    if rj is None:
        raise RuntimeError(f"RESULT_JSON なし stderr={(p.stderr or '')[-800:]}")
    return rj, elapsed


def treg_predict(name, strategy, test_features_csv, target):
    """trained_model を専用 env に固めて predict_template で予測。予測配列を返す。"""
    env = os.path.join(bc.WORK_DIR, name, f"penv_{strategy}")
    shutil.rmtree(env, ignore_errors=True)
    os.makedirs(env)
    shutil.copy(PREDICT_TEMPLATE, env)
    if os.path.exists(LIGHT):
        shutil.copy(LIGHT, env)
    shutil.copytree(TRAINED, os.path.join(env, "trained_model"))
    local_in = os.path.join(env, "test_features.csv")
    shutil.copy(test_features_csv, local_in)
    p = subprocess.run([EMBED_PY, os.path.join(env, "predict_template.py"), local_in],
                       capture_output=True, timeout=600, encoding="utf-8", errors="replace")
    out_csv = os.path.join(env, "test_features_predicted.csv")
    if not os.path.exists(out_csv):
        raise RuntimeError(f"予測出力なし stdout={(p.stdout or '')[-400:]} stderr={(p.stderr or '')[-400:]}")
    dfp = bc.read_any_csv(out_csv)
    return dfp[target].values


def run_one(name, meta):
    csv_path = os.path.join(bc.DATA_DIR, meta["csv"])
    target = meta["target"]
    df = bc.read_any_csv(csv_path)
    paths, n_tr, n_te = bc.write_split(name, df, target)
    test_df = bc.read_any_csv(paths["test"])
    y_test = test_df[target].values

    rec = {"dataset": name, "target": target, "n_train": n_tr, "n_test": n_te,
           "family": meta.get("family"), "ceiling_r2": meta.get("ceiling_r2"),
           "source": meta.get("source"), "note": meta.get("note"), "modes": {}}

    for strategy in ("quick", "thorough"):
        try:
            rj, elapsed = treg_train(paths["train"], target, strategy)
            preds = treg_predict(name, strategy, paths["test_features"], target)
            r2, rmse, mae, n_used = bc.score(y_test, preds)
            rec["modes"][strategy] = {
                "test_r2": r2, "test_rmse": rmse, "test_mae": mae, "n_scored": n_used,
                "self_r2": rj.get("r2"), "self_r2_std": rj.get("r2_std"),
                "train_r2": rj.get("train_r2"),
                "best_model": rj.get("best_model"), "model_type": rj.get("model_type"),
                "train_sec": round(elapsed, 1),
                "data_warning": rj.get("data_warning") or "",
            }
            print(f"  [{strategy:8s}] test_r2={r2}  self_r2={rj.get('r2')}  "
                  f"model={rj.get('best_model')}  {elapsed:.1f}s")
        except Exception as e:
            rec["modes"][strategy] = {"error": str(e)[:500]}
            print(f"  [{strategy:8s}] ERROR: {str(e)[:200]}")

    bc.append_jsonl(TREG_RECORDS, rec)


def main():
    manifests = sys.argv[1:] or ["manifest_synth.json", "manifest_real.json"]
    manifest = bc.load_manifest(*manifests)
    done = {r["dataset"] for r in bc.read_jsonl(TREG_RECORDS)}
    backup_trained_model()
    todo = [(n, m) for n, m in manifest.items() if n not in done]
    print(f"[treg] 対象 {len(todo)} 問(スキップ済 {len(done)}) manifests={manifests}")
    for i, (name, meta) in enumerate(todo, 1):
        print(f"[treg] ({i}/{len(todo)}) {name}  n={meta.get('n')} ceiling={meta.get('ceiling_r2')}")
        run_one(name, meta)
    print(f"[treg] 完了。結果 -> {TREG_RECORDS}")


if __name__ == "__main__":
    main()
