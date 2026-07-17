# 回帰スイート: quick不変 + thorough改善(外部テストR²のground truth) + native parity + エッジ
# 実行:  <ルート>\dist_portable\T-regressor\python-embed\python.exe tests\verify_rebuild.py
import sys, os, subprocess, shutil, json, time, struct
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tests/ の親 = プロジェクトルート
sys.path.insert(0, ROOT)
from _light import r2_score  # sklearn 非依存（本体と同じ自前実装）
# TREG_PYTHON: CI等でembed pythonの配置場所が異なる場合にハードコードを上書きできるように
# する(タスク6)。未設定時は従来通りdist_portable配下のembed pythonを既定値として使う。
PY = os.environ.get("TREG_PYTHON") or os.path.join(ROOT, r"dist_portable\T-regressor\python-embed\python.exe")
NATIVE = os.path.join(ROOT, r"native_predictor\predict_native.exe")
SP = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(SP, "verify_work")
shutil.rmtree(WORK, ignore_errors=True); os.makedirs(WORK)

FAILS = []

def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}: {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        FAILS.append(f"{name} {detail}")

def make_data(kind, n, seed):
    rng = np.random.RandomState(seed)
    if kind == "easy":
        X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(1, 9)})
        y = 2*X.x1 - X.x2 + rng.randn(n)*0.3
    elif kind == "hard":
        X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(1, 9)})
        y = (np.sin(X.x1*2)*3 + X.x2*X.x3 + np.where(X.x4 > 0, 2, -2)
             + 0.5*X.x5**2 + rng.randn(n)*1.5)
    elif kind == "smooth":  # GP向き: 少数特徴の滑らか関数
        X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(1, 4)})
        y = np.sin(X.x1) + 0.5*np.cos(X.x2*1.5) + 0.3*X.x3 + rng.randn(n)*0.1
    elif kind == "noise":
        X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(1, 6)})
        y = rng.randn(n)
    X = X.copy(); X["y"] = y
    return X

def train(train_csv, strategy, timeout=1200):
    t0 = time.time()
    p = subprocess.run([PY, os.path.join(ROOT, "train_bridge.py"), train_csv, "y", "0", strategy, "4"],
                       capture_output=True, cwd=ROOT, timeout=timeout, encoding="utf-8", errors="replace")
    elapsed = time.time() - t0
    out = p.stdout or ""
    rj = None
    for line in out.splitlines():
        if line.startswith("RESULT_JSON:"):
            rj = json.loads(line[len("RESULT_JSON:"):])
    return rj, elapsed, out

def py_predict(test_df):
    env = os.path.join(WORK, "_penv"); shutil.rmtree(env, ignore_errors=True); os.makedirs(env)
    shutil.copy(os.path.join(ROOT, "predict_template.py"), env)
    # _light.py も同梱（本番distと同じく predict_template.py の隣に置く）
    _lp = os.path.join(ROOT, "_light.py")
    if os.path.exists(_lp):
        shutil.copy(_lp, env)
    shutil.copytree(os.path.join(ROOT, "trained_model"), os.path.join(env, "trained_model"))
    test_in = os.path.join(WORK, "_ext_test.csv")
    for f in os.listdir(WORK):
        if f.startswith("_ext_test") and f != "_ext_test.csv":
            os.remove(os.path.join(WORK, f))
    test_df.drop(columns=["y"]).to_csv(test_in, index=False)
    p = subprocess.run([PY, os.path.join(env, "predict_template.py"), test_in],
                       capture_output=True, timeout=300, encoding="utf-8", errors="replace")
    out_csv = os.path.join(WORK, "_ext_test_predicted.csv")
    if not os.path.exists(out_csv):
        return None, (p.stdout or "") + (p.stderr or "")
    pred = pd.read_csv(out_csv)
    return pred["y"].values, p.stdout or ""

def native_predict(test_df, tag):
    treg = os.path.join(ROOT, "trained_model", "model.treg")
    csv_in = os.path.join(WORK, f"_nat_{tag}.csv")
    out_csv = os.path.join(WORK, f"_nat_{tag}_pred.csv")
    if os.path.exists(out_csv):
        os.remove(out_csv)
    test_df.drop(columns=["y"]).to_csv(csv_in, index=False)
    proc = subprocess.Popen([NATIVE, csv_in, treg])
    for _ in range(120):
        if os.path.exists(out_csv):
            time.sleep(0.5)
            break
        time.sleep(0.25)
    try:
        proc.kill()
    except Exception:
        pass
    if not os.path.exists(out_csv):
        return None
    return pd.read_csv(out_csv).iloc[:, -1].values

def treg_header():
    with open(os.path.join(ROOT, "trained_model", "model.treg"), "rb") as f:
        b = f.read(10)
    return b[4], b[5]  # version, type

TYPE_NAMES = {0: "linear", 1: "lgbm", 2: "gp", 3: "mlp", 4: "linear_poly", 5: "blend"}
NATIVE_TYPES = {0, 1, 2, 3, 4, 5}  # predict_native.exe (C++) が読める .treg モデル型

# ═══ 1. quick 回帰チェック (easy300) ═══════════════════════════════════════════
print("■ 1. quick 回帰チェック (easy300) — 挙動不変の確認", flush=True)
tr = make_data("easy", 300, 0); te = make_data("easy", 400, 99)
csv = os.path.join(WORK, "easy300.csv"); tr.to_csv(csv, index=False)
rj, el, out = train(csv, "quick")
check("RESULT_JSONパース", rj is not None)
check("quick時間 < 15s", el < 15, f"{el:.1f}s")
check("quickにFEが走らない", "[FE]" not in out)
ver, mtype = treg_header()
check("quickのtregはv3", ver == 3, f"version={ver}")
preds, _ = py_predict(te)
r2_q_easy = r2_score(te["y"], preds) if preds is not None else -9
check("quick easy 外部R² > 0.95", r2_q_easy > 0.95, f"R²={r2_q_easy:.4f}")

# ═══ 2. quick vs thorough (hard600) — 本丸 ═══════════════════════════════════
print("■ 2. hard600: quick vs thorough(新) の外部テストR²", flush=True)
tr = make_data("hard", 600, 0); te = make_data("hard", 1000, 99)
csv = os.path.join(WORK, "hard600.csv"); tr.to_csv(csv, index=False)

rj_q, el_q, _ = train(csv, "quick")
preds_q, _ = py_predict(te)
r2_q = r2_score(te["y"], preds_q) if preds_q is not None else -9

rj_t, el_t, out_t = train(csv, "thorough")
check("thorough RESULT_JSON", rj_t is not None)
check("FE実行ログあり", "[FE] 自動特徴量" in out_t)
check("LGBMランダムサーチログあり", "ランダムサーチ" in out_t)
check("LGBM-RF OOFログあり", "[LGBM-RF] OOF" in out_t)
check("LGBM-XT OOFログあり", "[LGBM-XT] OOF" in out_t)
preds_t, pt_out = py_predict(te)
check("thorough予測成功", preds_t is not None, pt_out[:200] if preds_t is None else "")
r2_t = r2_score(te["y"], preds_t) if preds_t is not None else -9
ver_t, mtype_t = treg_header()
print(f"  [結果] quick: R²={r2_q:.4f} ({el_q:.1f}s, {rj_q['best_model']})", flush=True)
print(f"  [結果] thorough: R²={r2_t:.4f} ({el_t:.1f}s, {rj_t['best_model']}) treg v{ver_t}/{TYPE_NAMES.get(mtype_t)}", flush=True)
check("thorough外部R²がquick+0.02以上", r2_t >= r2_q + 0.02, f"Δ={r2_t - r2_q:+.4f}")
print(f"  INFO: thorough時間 {el_t:.1f}s（負荷変動あり・参考値。合否には使わない）", flush=True)

# native parity (in-appモデルとdeployモデルが同型の場合のみ直接比較)
if mtype_t not in NATIVE_TYPES:
    print(f"  [情報] deployモデル型={TYPE_NAMES.get(mtype_t, mtype_t)} はpredict_native.exe未対応"
          f"（type4=linear_poly/type5=blend）のためnative検証をスキップします", flush=True)
else:
    nat = native_predict(te, "hard")
    check("native予測成功", nat is not None)
    if nat is not None and rj_t["model_type"] == TYPE_NAMES.get(mtype_t):
        diff = np.max(np.abs(nat - preds_t))
        rel = diff / max(np.std(te["y"].values), 1e-9)
        check("native parity (hard, FEあり)", rel < 2e-3, f"maxdiff={diff:.5f} rel={rel:.5f}")
    elif nat is not None:
        r2_nat = r2_score(te["y"], nat)
        print(f"  [情報] in-app={rj_t['model_type']} ≠ deploy={TYPE_NAMES.get(mtype_t)} → parityはR²健全性のみ: native R²={r2_nat:.4f}", flush=True)
        check("native R²健全 (hard)", r2_nat > max(0.3, r2_q - 0.1), f"R²={r2_nat:.4f}")

# ═══ 3. smooth150 thorough — GP系デプロイ + FE + native ═══════════════════════
print("■ 3. smooth150 thorough (GP向き小データ)", flush=True)
tr = make_data("smooth", 150, 0); te = make_data("smooth", 500, 99)
csv = os.path.join(WORK, "smooth150.csv"); tr.to_csv(csv, index=False)
rj_s, el_s, out_s = train(csv, "thorough")
check("smooth thorough完了", rj_s is not None, f"{el_s:.1f}s")
preds_s, _ = py_predict(te)
r2_s = r2_score(te["y"], preds_s) if preds_s is not None else -9
ver_s, mtype_s = treg_header()
print(f"  [結果] R²={r2_s:.4f} ({el_s:.1f}s, {rj_s['best_model']}) treg v{ver_s}/{TYPE_NAMES.get(mtype_s)}", flush=True)
check("smooth 外部R² > 0.9", r2_s > 0.9, f"R²={r2_s:.4f}")
if mtype_s not in NATIVE_TYPES:
    print(f"  [情報] deployモデル型={TYPE_NAMES.get(mtype_s, mtype_s)} はpredict_native.exe未対応"
          f"（type4=linear_poly/type5=blend）のためnative検証をスキップします", flush=True)
else:
    nat_s = native_predict(te, "smooth")
    if nat_s is not None and preds_s is not None and rj_s["model_type"] == TYPE_NAMES.get(mtype_s):
        diff = np.max(np.abs(nat_s - preds_s))
        rel = diff / max(np.std(te["y"].values), 1e-9)
        check("native parity (smooth)", rel < 2e-3, f"maxdiff={diff:.5f} rel={rel:.5f}")
    elif nat_s is not None:
        r2_nat = r2_score(te["y"], nat_s)
        check("native R²健全 (smooth)", r2_nat > 0.8, f"in-app={rj_s['model_type']} deploy={TYPE_NAMES.get(mtype_s)} R²={r2_nat:.4f}")
    else:
        check("native予測成功 (smooth)", False)

# ═══ 4. エッジケース ═══════════════════════════════════════════════════════════
print("■ 4. エッジケース", flush=True)
# 4a. 15行 thorough (FEスキップ、クラッシュしない)
tr = make_data("easy", 15, 1)
csv = os.path.join(WORK, "tiny15.csv"); tr.to_csv(csv, index=False)
rj_e, el_e, out_e = train(csv, "thorough")
check("15行thoroughクラッシュしない", rj_e is not None)
check("15行でFEなし", "[FE] 自動特徴量" not in out_e)

# 4b. ノイズデータ (R²<0でも完走)
tr = make_data("noise", 200, 2)
csv = os.path.join(WORK, "noise200.csv"); tr.to_csv(csv, index=False)
rj_n, el_n, _ = train(csv, "thorough")
check("noiseデータ完走", rj_n is not None, f"R²={rj_n['r2'] if rj_n else '?'}")

# 4c. 整数ターゲット + 歪みy (log1p+smear+round + FE + native)
rng = np.random.RandomState(5)
X = pd.DataFrame({f"x{i}": rng.randn(300) for i in range(1, 6)})
y = np.round(np.exp(1.0 + 0.8*X.x1 + 0.5*X.x1*X.x2 + rng.randn(300)*0.4)).astype(int)
X["y"] = y
csv = os.path.join(WORK, "intskew.csv"); X.to_csv(csv, index=False)
rj_i, el_i, out_i = train(csv, "thorough")
check("整数歪みy thorough完走", rj_i is not None, f"{rj_i['best_model'] if rj_i else '?'}")
Xte = pd.DataFrame({f"x{i}": rng.randn(300) for i in range(1, 6)})
Xte["y"] = 0
preds_i, _ = py_predict(Xte)
check("整数y予測が整数", preds_i is not None and np.all(np.mod(preds_i, 1.0) == 0))
ver_i, mtype_i = treg_header()
if mtype_i not in NATIVE_TYPES:
    print(f"  [情報] deployモデル型={TYPE_NAMES.get(mtype_i, mtype_i)} はpredict_native.exe未対応"
          f"（type4=linear_poly/type5=blend）のためnative検証をスキップします", flush=True)
else:
    nat_i = native_predict(Xte, "int")
    if nat_i is not None and rj_i["model_type"] == TYPE_NAMES.get(mtype_i):
        diff = np.max(np.abs(nat_i - preds_i))
        check("native parity (int/smear/round)", diff <= 1.0, f"maxdiff={diff:.4f}")
    elif nat_i is not None:
        check("native整数出力", np.all(np.mod(nat_i, 1.0) == 0))

print("\n═══ 結果 ═══", flush=True)
if FAILS:
    print(f"FAIL {len(FAILS)} 件:", flush=True)
    for f in FAILS:
        print(f"  - {f}", flush=True)
    sys.exit(1)
print("ALL PASS", flush=True)
