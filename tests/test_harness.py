# T-regressor 抜本改修の統合テストハーネス
# python-embed で実行する。train_bridge.py をサブプロセス起動し、RESULT_JSON・.treg・
# native exe・predict_template の整合を検証する。
import sys, os, json, struct, subprocess, shutil, tempfile, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tests/ の親 = プロジェクトルート
sys.path.insert(0, ROOT)  # pkl 内 _light オブジェクトの unpickle 用（直接 pickle.load する箇所がある）
# TREG_PYTHON: CI等でembed pythonの配置場所が異なる場合にハードコードを上書きできるように
# する(タスク6)。未設定時は従来通りdist_portable配下のembed pythonを既定値として使う。
PY     = os.environ.get("TREG_PYTHON") or os.path.join(ROOT, r"dist_portable\T-regressor\python-embed\python.exe")
TRAIN  = os.path.join(ROOT, "train_bridge.py")
NATIVE = os.path.join(ROOT, r"native_predictor\predict_native.exe")
MODEL_DIR = os.path.join(ROOT, "trained_model")
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_work_harness")  # 生成物を隔離
shutil.rmtree(SP, ignore_errors=True); os.makedirs(SP)

# 低-M16: quickモードでも use_poly は行数/特徴数だけで決まる(train_bridge.py参照)ため
# 小規模データでは linear_poly(type4)、thoroughならblend(type5)もデプロイされ得る。
# 固定辞書の直接インデックスだとその場合KeyErrorでハーネス自体が異常終了していた。
# T3のif rc==0ブロックが実行されない場合でもT4側から参照できるよう、モジュール
# トップレベルで定義しておく。
TYPE_NAMES = {0: "linear", 1: "lgbm", 2: "gp", 3: "mlp", 4: "linear_poly", 5: "blend"}

PASS, FAIL = [], []

def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS: {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL: {name}  {detail}")

def run_train(csv, target, strategy, timeout=600):
    p = subprocess.run([PY, TRAIN, csv, target, "0", strategy, "4"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=timeout, cwd=ROOT)
    out = p.stdout or ""
    rj = None
    for line in out.splitlines():
        if line.startswith("RESULT_JSON:"):
            rj = json.loads(line[len("RESULT_JSON:"):])
            # 注意: json.loadsはNaN/Infinityをデフォルトで受理するため、上の行は
            # 「構文的にパース可能」の確認であり、NaN/Inf混入の検出にはならない
            # (NaN/Inf自体の排除はtrain_bridge.py側のisfiniteガードで行っている)。
    return p.returncode, out, (p.stderr or ""), rj

def run_predict_template(csv):
    # trained_model の隣に predict_template を置いて実行
    tmp = os.path.join(SP, "_predict_env")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    shutil.copy(os.path.join(ROOT, "predict_template.py"), tmp)
    _lp = os.path.join(ROOT, "_light.py")
    if os.path.exists(_lp):
        shutil.copy(_lp, tmp)  # pkl 内 _light オブジェクトの unpickle 用
    shutil.copytree(MODEL_DIR, os.path.join(tmp, "trained_model"))
    p = subprocess.run([PY, os.path.join(tmp, "predict_template.py"), csv],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=300)
    pj = None
    for line in (p.stdout or "").splitlines():
        if line.startswith("PREDICT_JSON:"):
            pj = json.loads(line[len("PREDICT_JSON:"):])
    return p.returncode, p.stdout or "", pj

def wait_for_stable_file(path, poll_interval=0.5, max_iters=60):
    # Low C-3: 固定0.5秒sleepだと、native exeの書き込みが遅い環境で部分書き込みの
    # CSVを読んでしまうレースがあり得た。ファイルサイズが2回連続で同じ値(かつ非ゼロ)
    # になるまでポーリングして、書き込み完了を実際に確認する。
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

def run_native(csv):
    out_csv = os.path.splitext(csv)[0] + "_pred.csv"
    if os.path.exists(out_csv):
        os.remove(out_csv)
    treg = os.path.join(MODEL_DIR, "model.treg")
    # GUI exe なのでメッセージボックスが出る → タイムアウト付きで起動し、出力生成を待って kill
    proc = subprocess.Popen([NATIVE, csv, treg])
    wait_for_stable_file(out_csv)
    try:
        proc.kill()
    except Exception as e:
        # 以前はkill失敗を無条件で無視しており、原因不明のテスト不安定化の手がかりが
        # 消えていた(Low C-3)。失敗時はログだけ残して処理は継続する。
        print(f"  [警告] native process kill失敗: {e}")
    if not os.path.exists(out_csv):
        return None
    return pd.read_csv(out_csv)

def read_treg_header(path):
    with open(path, "rb") as f:
        data = f.read()
    return data[4], data[5], struct.unpack("<I", data[6:10])[0]  # version, type, n_feat


# ══════════════════════════════════════════════════════════════════════
print("=== T1: 400行通常データ quick ===")
np.random.seed(0)
n = 400
d1 = pd.DataFrame({
    "x1": np.random.randn(n), "x2": np.random.randn(n), "x3": np.random.randn(n),
    "cat": np.random.choice(["a","b","c"], n),
})
d1["y"] = 3*d1.x1 - 2*d1.x2 + 0.5*d1.x3**2 + d1.cat.map({"a":0,"b":2,"c":-1}) + np.random.randn(n)*0.5
d1["y"] = np.abs(d1["y"]) + 1  # 歪み → log1p 誘発
csv1 = os.path.join(SP, "t1.csv"); d1.to_csv(csv1, index=False)
rc, out, err, rj = run_train(csv1, "y", "quick")
check("T1 quick 完走", rc == 0, f"rc={rc} err_tail={err[-300:]}")
check("T1 RESULT_JSON strict parse", rj is not None)
if rj:
    check("T1 r2 と r2_raw が存在", "r2" in rj and "r2_raw" in rj)
    check("T1 r2 > 0.5", (rj.get("r2") or 0) > 0.5, f"r2={rj.get('r2')}")
check("T1 model.treg 生成", os.path.exists(os.path.join(MODEL_DIR, "model.treg")))
check("T1 tmpディレクトリ残置なし", not os.path.exists(os.path.join(ROOT, "trained_model_tmp")))

print("=== T2: 同データ thorough (Blend OOF + parity) ===")
rc, out, err, rj2 = run_train(csv1, "y", "thorough")
check("T2 thorough 完走", rc == 0, f"rc={rc}")
check("T2 OOF R²ログ (全モデル)", ("OOF R²" in out))
if rj2:
    print(f"    best={rj2['best_model']} r2={rj2['r2']} r2_raw={rj2['r2_raw']} eval_on={rj2['eval_on']}")
    check("T2 eval_on=oof", rj2.get("eval_on") == "oof")

# Blend が best の場合: in-app予測が学習時 OOF blend と整合するか
if rj2 and rj2.get("model_type") == "blend":
    # 学習データ自身を予測 → blend生重み内積の再現をチェック
    pred_input = os.path.join(SP, "t2_pred.csv")
    d1.drop(columns=["y"]).to_csv(pred_input, index=False)
    prc, pout, pj = run_predict_template(pred_input)
    check("T2 blend in-app予測 完走", prc == 0 and pj is not None, f"rc={prc}")
    if pj:
        check("T2 missing_cols フィールド存在", "missing_cols" in pj and pj["missing_cols"] == [])
else:
    print("    (blend が best でないため in-app blend parity はスキップ)")

print("=== T3: 15特徴量スクリーニング + GP/MLP deploy 次元整合 ===")
np.random.seed(7)
n = 250
cols = {f"s{i}": np.random.randn(n) for i in range(1, 6)}
cols.update({f"noise{i}": np.random.randn(n) for i in range(1, 11)})
d3 = pd.DataFrame(cols)
d3["y"] = 2*d3.s1 - 1.5*d3.s2 + np.sin(d3.s3*2) + 0.5*d3.s4*d3.s5 + np.random.randn(n)*0.3
csv3 = os.path.join(SP, "t3.csv"); d3.to_csv(csv3, index=False)
rc, out, err, rj3 = run_train(csv3, "y", "quick")
check("T3 quick 完走", rc == 0)
if rc == 0:
    ver, mtype, n_feat = read_treg_header(os.path.join(MODEL_DIR, "model.treg"))
    # deploy されたモデルの実使用列数を pkl/meta から取得
    import pickle
    actual_dim = None
    tm = TYPE_NAMES.get(mtype, f"unknown(type{mtype})")
    if tm == "gp":
        with open(os.path.join(MODEL_DIR, "gp_model.pkl"), "rb") as f:
            actual_dim = len(pickle.load(f)["feat_cols"])
    elif tm == "mlp":
        with open(os.path.join(MODEL_DIR, "mlp_model.pkl"), "rb") as f:
            actual_dim = len(pickle.load(f)["feat_cols"])
    elif tm == "lgbm":
        with open(os.path.join(MODEL_DIR, "lgbm_meta.json"), encoding="utf-8") as f:
            actual_dim = len(json.load(f)["feat_cols"])
    elif tm == "linear" or tm == "linear_poly":
        # poly-Ridgeの特徴情報もlinear_model.pklに同梱される(use_poly=Trueのケース)
        with open(os.path.join(MODEL_DIR, "linear_model.pkl"), "rb") as f:
            actual_dim = len(pickle.load(f)["feat_cols"])
    elif tm == "blend":
        # blendは複数サブモデルの合成で単一のn_feat比較にはなじまない(.tregのn_feat=0が
        # 正当な値、predict-core.js/treg-writer参照)ため次元チェック自体をスキップする。
        pass
    print(f"    deploy_type={tm} treg_n_feat={n_feat} actual_dim={actual_dim}")
    if tm != "blend":
        check("T3 .treg n_feat == モデル実次元", n_feat == actual_dim, f"{n_feat} != {actual_dim}")

    # native parity: predict_native.exeはtype0-5(linear_poly/blend含む)全対応
    pred_in3 = os.path.join(SP, "t3_pred_in.csv")
    d3.drop(columns=["y"]).head(30).to_csv(pred_in3, index=False)
    ndf = run_native(pred_in3)
    prc, _, pj3 = run_predict_template(pred_in3)
    check("T3 native 実行", ndf is not None)
    check("T3 python 予測実行", prc == 0)
    if ndf is not None and prc == 0:
        py_out = pd.read_csv(os.path.join(SP, "_predict_env") + r"\..\t3_pred_in_predicted.csv") \
            if False else pd.read_csv(os.path.join(SP, "t3_pred_in_predicted.csv"))
        diff = np.abs(ndf["y"].values - py_out["y"].values).max()
        # Low C-3: この閾値(1e-2)はtests/verify_rebuild.pyの2e-3と一見矛盾するが、
        # 正規化の基準が異なるため単純に数値だけ揃えても意味が変わる。ここは
        # native予測自身の平均絶対値で正規化(分母が小さくなりがちで緩めの閾値になる)、
        # verify_rebuild.py側は外部テストyの標準偏差で正規化(分母が通常大きく、
        # 厳しめの閾値になる)。詳細はtests/verify_rebuild.pyの同種コメント参照。
        rel = diff / max(1e-9, np.abs(py_out["y"].values).mean())
        print(f"    native vs python max diff = {diff:.5f} (rel {rel:.5f})")
        check("T3 native/python parity", rel < 1e-2, f"rel={rel}")

print("=== T4: LGBM-best データ (C1回帰) ===")
np.random.seed(1)
n = 600
d4 = pd.DataFrame({f"x{i}": np.random.randn(n) for i in range(1, 6)})
# 階段関数 + 交互作用 → 木モデル有利
d4["y"] = (np.where(d4.x1 > 0, 5, 0) + np.where(d4.x2 > 0.5, 3, -1)
           + np.where((d4.x3 > 0) & (d4.x4 > 0), 4, 0) + np.random.randn(n)*0.2)
csv4 = os.path.join(SP, "t4.csv"); d4.to_csv(csv4, index=False)
rc, out, err, rj4 = run_train(csv4, "y", "quick")
check("T4 quick 完走", rc == 0)
if rj4:
    print(f"    best={rj4['best_model']} r2={rj4['r2']}")
if rc == 0:
    ver, mtype, n_feat = read_treg_header(os.path.join(MODEL_DIR, "model.treg"))
    # 低-M16: T3と同じくquickモードでもuse_poly/blendになり得るため固定辞書直索引は避ける
    tm = TYPE_NAMES.get(mtype, f"unknown(type{mtype})")
    print(f"    deploy_type={tm}")
    # native parity: predict_native.exeはtype0-5(linear_poly/blend含む)全対応
    pred_in4 = os.path.join(SP, "t4_pred_in.csv")
    d4.drop(columns=["y"]).head(30).to_csv(pred_in4, index=False)
    ndf = run_native(pred_in4)
    prc, _, pj4 = run_predict_template(pred_in4)
    if ndf is not None and prc == 0:
        py_out = pd.read_csv(os.path.join(SP, "t4_pred_in_predicted.csv"))
        diff = np.abs(ndf["y"].values - py_out["y"].values).max()
        # 閾値1e-2の根拠は上のT3ブロックのコメント参照(正規化基準の違い)。
        rel = diff / max(1e-9, np.abs(py_out["y"].values).mean())
        print(f"    native vs python max diff = {diff:.5f} (rel {rel:.5f})")
        check("T4 native/python parity", rel < 1e-2, f"rel={rel}")
        # 定数出力になっていないこと（C1の症状）
        check("T4 native出力が定数でない", ndf["y"].std() > 0.1, f"std={ndf['y'].std()}")
    else:
        check("T4 native+python 実行", False, f"native={ndf is not None} py={prc}")

print("=== T5: エッジケース ===")
# 5a: 15行
d5a = pd.DataFrame({"a": np.random.randn(15), "b": np.random.randn(15)})
d5a["y"] = d5a.a * 2 + np.random.randn(15)*0.1
csv5a = os.path.join(SP, "t5a.csv"); d5a.to_csv(csv5a, index=False)
rc, out, err, rj = run_train(csv5a, "y", "quick")
check("T5a 15行で完走", rc == 0 and rj is not None, f"rc={rc}")

# 5b: target NaN 混入
d5b = d1.copy()
d5b.loc[d5b.index[:20], "y"] = np.nan
csv5b = os.path.join(SP, "t5b.csv"); d5b.to_csv(csv5b, index=False)
rc, out, err, rj = run_train(csv5b, "y", "quick")
check("T5b target NaN で完走+警告", rc == 0 and rj is not None and "除外" in (rj.get("data_warning") or ""),
      f"rc={rc} warn={rj.get('data_warning') if rj else None}")

# 5c: 文字列target
d5c = d1.copy(); d5c["y"] = "abc"
csv5c = os.path.join(SP, "t5c.csv"); d5c.to_csv(csv5c, index=False)
rc, out, err, rj = run_train(csv5c, "y", "quick")
check("T5c 文字列targetで明示エラー", rc == 1 and "数値でない" in out, f"rc={rc}")

# 5d: 存在しないtarget名
rc, out, err, rj = run_train(csv1, "no_such_col", "quick")
check("T5d 不在target名で明示エラー", rc == 1 and "存在しません" in out, f"rc={rc}")

# 5e: 整数target丸め
d5e = d1.copy(); d5e["y"] = np.round(d5e["y"]).astype(int)
csv5e = os.path.join(SP, "t5e.csv"); d5e.to_csv(csv5e, index=False)
rc, out, err, rj = run_train(csv5e, "y", "quick")
ok_round = False
if rc == 0:
    with open(os.path.join(MODEL_DIR, "model_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    ok_round = meta.get("postprocess", {}).get("round_output") is True
check("T5e 整数target で round_output=True", rc == 0 and ok_round)

# 5f: 12行 (旧: StratifiedKFold全滅ケース)
d5f = pd.DataFrame({"a": np.random.randn(12), "b": np.random.randn(12)})
d5f["y"] = d5f.a * 2 + np.random.randn(12)*0.1
csv5f = os.path.join(SP, "t5f.csv"); d5f.to_csv(csv5f, index=False)
rc, out, err, rj = run_train(csv5f, "y", "quick")
check("T5f 12行で完走 (旧全滅ケース)", rc == 0 and rj is not None, f"rc={rc} out_tail={out[-200:]}")

print()
print("=" * 60)
print(f"PASS: {len(PASS)}  FAIL: {len(FAIL)}")
for name, detail in FAIL:
    print(f"  FAIL: {name}  {detail}")
sys.exit(0 if not FAIL else 1)
