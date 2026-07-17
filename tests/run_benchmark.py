# tests/run_benchmark.py — リリース前ベンチマークスイート。
#
# tests/benchmarks/data/ の合成データセット群(tests/benchmarks/gen_datasets.py で生成)を
# quick/thorough両モードで学習し、{dataset, mode, best_model, model_type, r2, r2_std,
# train_r2, rmse, mae, elapsed_sec, data_warnings}をJSON記録する。学習済み.tregを
# C++(native_predictor)/JS(predict-core.js経由のrun_e2e_predict.js)/Python(predict_template.py)
# の3エンジンで予測し、tests/e2e_compare_predictions.pyで数値パリティを確認する。
#
# 実行(python-embed。lightgbm等の依存はtrain_bridge.py側がサブプロセスとして必要とする。
# 本スクリプト自身はpandasのみに依存):
#     <embed python> tests/run_benchmark.py                       # baseline比較(通常運用)
#     <embed python> tests/run_benchmark.py --update-baseline     # baseline.json更新(upsert)
#     <embed python> tests/run_benchmark.py --mode quick          # quickのみ(CI向け、高速)
#     <embed python> tests/run_benchmark.py --datasets linear_clean,pure_noise
#
# 環境変数 TREG_PYTHON / TREG_NATIVE / TREG_NODE で使用するpython.exe/predict_native.exe/node
# を上書きできる(verify_rebuild.pyと同じ慣習)。
#
# 合否基準:
#   - baseline比でR²が0.02超低下 → FAIL
#   - pure_noiseのR² > 0.15(リーク検知番犬) → FAIL
#   - best_modelの種別(model_type)がbaselineと変化 → WARN(FAILにはしない)
#   - 3エンジン(C++/JS/Python)の予測パリティ不一致 → FAIL
#   - 各データセット固有の期待特性チェック(dataset_expectations) → FAIL
#   - 所要時間(elapsed_sec)は記録するがINFO表示のみで合否には使わない
#
# 注: quick/thoroughは内部の評価手法(eval_on)が異なる場合があり(例: thoroughは明示的に
# 「OOF(交差検証)で評価」と表示することがある)、自己申告R²同士を安易にモード間で比較すると
# 手法差を「劣化」と誤検知しうる。そのため本スイートはモード間比較を行わず、
# 同一モード同士(quick vs quick、thorough vs thorough)のbaseline比較に限定している。
#
# 終了コード: 全PASS(WARNのみ許容)なら0、1件でもFAILがあれば1。
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tests/ の親 = プロジェクトルート

BENCH_DIR = os.path.join(ROOT, "tests", "benchmarks")
DATA_DIR = os.path.join(BENCH_DIR, "data")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")
BASELINE_PATH = os.path.join(BENCH_DIR, "baseline.json")
WORK = os.path.join(BENCH_DIR, "_work")

PY = os.environ.get("TREG_PYTHON") or os.path.join(ROOT, r"dist_portable\T-regressor\python-embed\python.exe")
NATIVE = os.environ.get("TREG_NATIVE") or os.path.join(ROOT, r"native_predictor\predict_native.exe")
NODE = os.environ.get("TREG_NODE") or "node"
RUN_E2E_JS = os.path.join(ROOT, "web", "js_predict_poc", "run_e2e_predict.js")
COMPARE_PY = os.path.join(ROOT, "tests", "e2e_compare_predictions.py")

R2_REGRESSION_THRESHOLD = 0.02   # baseline比R²低下でFAILにする閾値
PURE_NOISE_R2_MAX = 0.15         # リーク検知番犬: これを超えたらFAIL
TRAIN_TIMEOUT = 1800
TYPE_NAMES = {0: "linear", 1: "lgbm", 2: "gp", 3: "mlp", 4: "linear_poly", 5: "blend"}


def log(msg):
    print(msg, flush=True)


def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        log(f"[FATAL] {MANIFEST_PATH} がありません。先に "
            f"`python tests/benchmarks/gen_datasets.py` を実行してください。")
        sys.exit(1)
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def train(csv_path, target_col, strategy, timeout=TRAIN_TIMEOUT):
    t0 = time.time()
    p = subprocess.run(
        [PY, os.path.join(ROOT, "train_bridge.py"), csv_path, target_col, "0", strategy, "4"],
        capture_output=True, cwd=ROOT, timeout=timeout, encoding="utf-8", errors="replace")
    elapsed = time.time() - t0
    out = p.stdout or ""
    rj = None
    for line in out.splitlines():
        if line.startswith("RESULT_JSON:"):
            rj = json.loads(line[len("RESULT_JSON:"):])
    return rj, elapsed, out, (p.stderr or "")


def treg_header():
    treg_path = os.path.join(ROOT, "trained_model", "model.treg")
    with open(treg_path, "rb") as f:
        b = f.read(10)
    return b[4], b[5]  # file_version, model_type


def wait_for_stable_file(path, poll_interval=0.25, max_iters=160):
    # verify_rebuild.pyのwait_for_stable_fileと同じ方針(Low C-3): ファイルサイズが
    # 2回連続で同じ非ゼロ値になるまでポーリングし、native exeの部分書き込みを読まない。
    last_size = None
    for _ in range(max_iters):
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size > 0 and size == last_size:
                return True
            last_size = size
        else:
            last_size = None
        time.sleep(poll_interval)
    return os.path.exists(path)


def make_predict_input(csv_path, target_col, tag, n_rows=20):
    """訓練CSVの先頭n_rows行からtarget列を落としたUTF-8 CSVを作る(3エンジンパリティ検証専用の
    入力)。元CSVがShift-JIS(jp_columns_sjis)であっても、ここではUTF-8に統一する。native/JSの
    Shift-JIS入力対応には既知の制約があり(README.md「書き出しEXEとウイルス対策ソフト」節とは
    別に、web/js_predict_poc/README.mdのE14参照)、ここでの検証目的は「.treg読み込み後の
    数値一致」であって入力エンコーディング自体の検証ではないため、UTF-8に揃えて本題を汚さない。
    """
    try:
        with open(csv_path, "rb") as f:
            raw = f.read()
        raw.decode("utf-8")
        enc = "utf-8"
    except UnicodeDecodeError:
        enc = "cp932"
    df = pd.read_csv(csv_path, encoding=enc, nrows=n_rows)
    df = df.drop(columns=[target_col])
    out_path = os.path.join(WORK, f"{tag}_predict_in.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def python_predict(tag):
    env = os.path.join(WORK, f"{tag}_penv")
    shutil.rmtree(env, ignore_errors=True)
    os.makedirs(env)
    shutil.copy(os.path.join(ROOT, "predict_template.py"), env)
    lp = os.path.join(ROOT, "_light.py")
    if os.path.exists(lp):
        shutil.copy(lp, env)
    shutil.copytree(os.path.join(ROOT, "trained_model"), os.path.join(env, "trained_model"))
    local_in = os.path.join(env, f"{tag}_predict_in.csv")
    shutil.copy(os.path.join(WORK, f"{tag}_predict_in.csv"), local_in)
    p = subprocess.run([PY, os.path.join(env, "predict_template.py"), local_in],
                        capture_output=True, timeout=300, encoding="utf-8", errors="replace")
    out_csv = os.path.join(env, f"{tag}_predict_in_predicted.csv")
    if not os.path.exists(out_csv):
        return None, (p.stdout or "") + (p.stderr or "")
    return out_csv, p.stdout or ""


def native_predict(tag):
    treg = os.path.join(ROOT, "trained_model", "model.treg")
    in_csv = os.path.join(WORK, f"{tag}_predict_in.csv")
    out_csv = os.path.join(WORK, f"{tag}_predict_in_pred.csv")  # predict_native_v2.cppの命名規則
    if os.path.exists(out_csv):
        os.remove(out_csv)
    proc = subprocess.Popen([NATIVE, in_csv, treg])
    wait_for_stable_file(out_csv)
    # GUIサブシステムのため完了後もメッセージボックスが残る(verify_rebuild.pyと同じ事情)。
    try:
        proc.kill()
    except Exception as e:
        log(f"    [警告] native process kill失敗(tag={tag}): {e}")
    return out_csv if os.path.exists(out_csv) else None


def js_predict(tag):
    treg = os.path.join(ROOT, "trained_model", "model.treg")
    in_csv = os.path.join(WORK, f"{tag}_predict_in.csv")
    out_csv = os.path.join(WORK, f"{tag}_predict_in_js.csv")
    if os.path.exists(out_csv):
        os.remove(out_csv)
    p = subprocess.run([NODE, RUN_E2E_JS, in_csv, treg, out_csv],
                        capture_output=True, timeout=120, encoding="utf-8", errors="replace")
    if not os.path.exists(out_csv):
        return None, (p.stdout or "") + (p.stderr or "")
    return out_csv, p.stdout or ""


# e2e_compare_predictions.pyの既定許容誤差(abs=1e-5, rel=1e-4)はlinear/lgbm/mlp等の
# 実測で決めたもので、GPモデルは.treg側がls(length-scale)をfloat32でしか保持できない
# ため、ARD-RBFカーネルのexp(-距離^2/ls^2)でfloat32丸め誤差が指数的に増幅されやすい
# (実測でpython(app)のpickle直読み vs .treg経由native/JSの間に最大1.4e-3程度のズレが
# 出ることを本スイートの実装時に確認済み。native/JS間は4.8e-7とほぼ完全一致するため
# .treg変換そのものは正しく、pickle(float64)と.treg(float32)の精度差が原因)。
# tests/verify_rebuild.pyも同種の理由でnative parityにrel<2e-3という緩い閾値を使っており
# (「丸めは1未満のノイズしか消さない」というdocs/treg-format.mdの教訓と同じ事情)、
# 本スイートもそれに合わせる。
PARITY_TOL_ABS = "1e-3"
PARITY_TOL_REL = "2e-3"


def check_parity(target_col, tag):
    """3エンジンの予測値パリティをtests/e2e_compare_predictions.py経由で確認する
    (既存の許容誤差ロジックを再利用。重複実装を避ける)。"""
    py_csv, py_log = python_predict(tag)
    if py_csv is None:
        return False, f"python予測に失敗: {py_log[-500:]}"
    nat_csv = native_predict(tag)
    if nat_csv is None:
        return False, "native予測に失敗(出力CSVが生成されなかった)"
    js_csv, js_log = js_predict(tag)
    if js_csv is None:
        return False, f"JS予測に失敗: {js_log[-500:]}"
    p = subprocess.run([PY, COMPARE_PY, target_col, py_csv, nat_csv, js_csv,
                        "--tol-abs", PARITY_TOL_ABS, "--tol-rel", PARITY_TOL_REL],
                        capture_output=True, timeout=120, encoding="utf-8", errors="replace")
    ok = p.returncode == 0
    return ok, (p.stdout or "") + (p.stderr or "")


def extract_best_candidate_field(rj, field):
    for c in rj.get("candidate_models", []):
        if c.get("is_best"):
            return c.get(field)
    return None


def dataset_expectations(name, rj, record):
    """データセット固有の期待特性チェック。(name, ok, detail)のリストを返す。
    しきい値は「回帰(退行)検出の番犬」であって精度追求のベンチマークではないため、
    こねくり回した特徴生成が偶然コケても過検知しない程度にゆるく取る。"""
    checks = []
    r2 = record["r2"] if record["r2"] is not None else float("-inf")
    dw = rj.get("data_warning") or ""
    cat_cols = set(rj.get("cat_columns") or [])
    cat_dropped = set(rj.get("cat_dropped_columns") or [])

    if name == "linear_clean":
        checks.append(("高R²(>0.85)", r2 > 0.85, f"r2={r2:.4f}"))
    elif name == "nonlinear_interaction":
        checks.append(("一定の説明力(r2>0.2)", r2 > 0.2, f"r2={r2:.4f}"))
    elif name == "categorical_low":
        checks.append(("grade列をカテゴリ検出(one-hot)", "grade" in cat_cols, f"cat_columns={sorted(cat_cols)}"))
        checks.append(("高R²(>0.7)", r2 > 0.7, f"r2={r2:.4f}"))
    elif name == "categorical_high":
        checks.append(("city列をカテゴリ検出(target encoding)", "city" in cat_cols, f"cat_columns={sorted(cat_cols)}"))
        checks.append(("raw_id列を高カーディナリティで除外", "raw_id" in cat_dropped, f"cat_dropped_columns={sorted(cat_dropped)}"))
    elif name == "bool_mixed":
        checks.append(("一定の説明力(r2>0.5)", r2 > 0.5, f"r2={r2:.4f}"))
    elif name == "missing_heavy":
        checks.append(("欠損があっても完走しR²>0.2", r2 > 0.2, f"r2={r2:.4f}"))
    elif name == "skewed_target":
        checks.append(("歪みyでも高R²(>0.6)", r2 > 0.6, f"r2={r2:.4f}"))
    elif name == "outlier_contaminated":
        checks.append(("外れ値警告を検出", "外れ値" in dw, f"data_warning={dw!r}"))
    elif name == "small_n":
        checks.append(("n=30でも完走", rj is not None, ""))
    elif name == "pure_noise":
        checks.append((f"R²<={PURE_NOISE_R2_MAX}(リーク検知番犬)", r2 <= PURE_NOISE_R2_MAX, f"r2={r2:.4f}"))
    elif name == "duplicated_rows":
        checks.append(("重複行警告を検出", "重複" in dw, f"data_warning={dw!r}"))
    elif name in ("jp_columns", "jp_columns_sjis"):
        checks.append(("地域列をカテゴリ検出(日本語列名/値)", "地域" in cat_cols, f"cat_columns={sorted(cat_cols)}"))
        checks.append(("完走しR²>0.3", r2 > 0.3, f"r2={r2:.4f}"))
    return checks


def run_one(name, meta, mode):
    csv_path = os.path.join(DATA_DIR, meta["csv"])
    target_col = meta["target"]
    tag = f"{name}_{mode}"
    log(f"\n■ {name} [{mode}] 学習中... (csv={meta['csv']}, target={target_col!r})")
    rj, elapsed, out, err = train(csv_path, target_col, mode)
    if rj is None:
        tail = (out[-500:] + err[-500:])
        log(f"  [FATAL] RESULT_JSONが得られませんでした。ログ末尾: {tail!r}")
        return {"dataset": name, "mode": mode, "fatal": True, "error": tail,
                "elapsed_sec": round(elapsed, 1)}

    ver, mtype = treg_header()
    record = {
        "dataset": name,
        "mode": mode,
        "best_model": rj.get("best_model"),
        "model_type": rj.get("model_type"),
        "r2": rj.get("r2"),
        "r2_std": extract_best_candidate_field(rj, "r2_std"),
        "train_r2": extract_best_candidate_field(rj, "train_r2"),
        "rmse": rj.get("rmse"),
        "mae": rj.get("mae"),
        "elapsed_sec": round(elapsed, 1),
        "data_warnings": rj.get("data_warning") or "",
        "treg_version": ver,
        "treg_model_type": TYPE_NAMES.get(mtype, mtype),
    }
    log(f"  [結果] best={record['best_model']}({record['model_type']}) r2={record['r2']} "
        f"rmse={record['rmse']} mae={record['mae']}")
    log(f"  INFO: 所要時間 {record['elapsed_sec']}s(負荷変動あり・参考値。合否には使わない)")

    try:
        make_predict_input(csv_path, target_col, tag)
        parity_ok, parity_log = check_parity(target_col, tag)
    except Exception as e:
        parity_ok, parity_log = False, f"パリティ検証中に例外: {e}"
    record["parity_ok"] = parity_ok
    if not parity_ok:
        log(f"  [FAIL] 3エンジン(C++/JS/Python)パリティ不一致:\n{parity_log[:800]}")
    else:
        log("  [OK] 3エンジン(C++/JS/Python)パリティ一致")

    checks = dataset_expectations(name, rj, record)
    record["dataset_checks"] = [{"name": n, "ok": ok, "detail": d} for (n, ok, d) in checks]
    for c in record["dataset_checks"]:
        log(f"  {'PASS' if c['ok'] else 'FAIL'}: {c['name']} ({c['detail']})")

    return record


def evaluate(records, baseline_by_key):
    """baseline_by_key=None なら退行比較(R²低下/model_type変化)は行わない
    (--update-baseline時: 学習/パリティ/データセット固有チェックの異常だけ検出する)。"""
    fails, warns = [], []
    for r in records:
        tag = f"{r['dataset']}[{r['mode']}]"
        if r.get("fatal"):
            fails.append(f"{tag}: 学習が完走しませんでした ({r.get('error', '')[:200]!r})")
            continue
        if not r.get("parity_ok", False):
            fails.append(f"{tag}: 3エンジンパリティ不一致")
        for c in r.get("dataset_checks", []):
            if not c["ok"]:
                fails.append(f"{tag}: データセット固有チェック失敗 [{c['name']}] {c['detail']}")
        if baseline_by_key is not None:
            base = baseline_by_key.get((r["dataset"], r["mode"]))
            if base is None:
                warns.append(f"{tag}: baselineに存在しない新規データセット/モード(退行比較スキップ)")
            else:
                r2, br2 = r.get("r2"), base.get("r2")
                if r2 is not None and br2 is not None:
                    drop = br2 - r2
                    if drop > R2_REGRESSION_THRESHOLD:
                        fails.append(f"{tag}: R²退行 baseline={br2:.4f} → 現在={r2:.4f} (Δ={-drop:+.4f})")
                if base.get("model_type") and r.get("model_type") and base["model_type"] != r["model_type"]:
                    warns.append(f"{tag}: best_model種別変化 {base['model_type']} → {r['model_type']}")
    return fails, warns


def print_summary(records, fails, warns):
    log("\n═══ ベンチマーク結果一覧 ═══")
    header = f"{'dataset':<24}{'mode':<10}{'best_model':<14}{'r2':>8}{'rmse':>10}{'elapsed_sec':>12}"
    log(header)
    for r in records:
        if r.get("fatal"):
            log(f"{r['dataset']:<24}{r['mode']:<10}{'(FATAL)':<14}{'':>8}{'':>10}{r.get('elapsed_sec',''):>12}")
            continue
        r2s = f"{r['r2']:.4f}" if r.get("r2") is not None else "?"
        rmses = f"{r['rmse']:.3f}" if r.get("rmse") is not None else "?"
        log(f"{r['dataset']:<24}{r['mode']:<10}{str(r['best_model'])[:13]:<14}{r2s:>8}{rmses:>10}{r['elapsed_sec']:>12}")

    log(f"\n═══ 判定: FAIL {len(fails)}件 / WARN {len(warns)}件 ═══")
    for w in warns:
        log(f"  [WARN] {w}")
    for f in fails:
        log(f"  [FAIL] {f}")
    if not fails and not warns:
        log("  ALL PASS")


def upsert_baseline(records):
    """既存baseline.jsonの(dataset,mode)キーを、今回学習した分だけ上書きする(upsert)。
    --datasets/--modeで絞り込んだ部分実行でも、他データセットの既存基準値を消さない
    (thoroughは実行時間が長いため、データセット単位で複数回に分けてbaselineを積み上げる
    運用を想定)。"""
    existing = {}
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH, encoding="utf-8") as f:
            old = json.load(f)
        for r in old.get("records", []):
            existing[(r["dataset"], r["mode"])] = r
    for r in records:
        if r.get("fatal"):
            continue  # 失敗した学習結果を基準値として固定しない
        existing[(r["dataset"], r["mode"])] = r
    merged = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "records": [existing[k] for k in sorted(existing.keys())],
    }
    with open(BASELINE_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return len(merged["records"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update-baseline", action="store_true", help="tests/benchmarks/baseline.jsonをupsert更新する")
    ap.add_argument("--mode", choices=["quick", "thorough", "both"], default="both")
    ap.add_argument("--datasets", default=None, help="カンマ区切りでデータセット名を絞り込む(省略時は全件)")
    args = ap.parse_args()

    manifest = load_manifest()
    names = list(manifest.keys())
    if args.datasets:
        want = set(args.datasets.split(","))
        missing = want - set(names)
        if missing:
            log(f"[警告] manifestに無いデータセット名を無視します: {sorted(missing)}")
        names = [n for n in names if n in want]
    if not names:
        log("[FATAL] 対象データセットが0件です。")
        sys.exit(1)

    modes = ["quick", "thorough"] if args.mode == "both" else [args.mode]

    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK, exist_ok=True)

    records = []
    for name in names:
        for mode in modes:
            records.append(run_one(name, manifest[name], mode))

    if args.update_baseline:
        # ベースライン更新モードでも、学習失敗・パリティ不一致・データセット固有チェックの
        # 異常は見逃さない(退行比較=baseline自己比較のみ対象外にする。異常値を基準として
        # 固定してしまう事故を防ぐため)。
        fails, warns = evaluate(records, baseline_by_key=None)
        n_total = upsert_baseline(records)
        print_summary(records, fails, warns)
        log(f"\nbaseline.json を更新しました(今回{len(records)}件 upsert、計{n_total}件)。")
        sys.exit(1 if fails else 0)

    if not os.path.exists(BASELINE_PATH):
        log("[FATAL] baseline.jsonがありません。初回は --update-baseline で生成してください。")
        sys.exit(1)
    with open(BASELINE_PATH, encoding="utf-8") as f:
        baseline = json.load(f)
    baseline_by_key = {(r["dataset"], r["mode"]): r for r in baseline.get("records", [])}

    fails, warns = evaluate(records, baseline_by_key)
    print_summary(records, fails, warns)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
