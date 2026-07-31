"""gen_datetime_fixtures.py — .treg v6 datetime_parts カテゴリエンコーダ
パリティフィクスチャ生成(2026-07第3弾・真因④対策「datetime列がID列扱いで破棄される」)。

train_bridge.py の内部関数(_prepare_categoricals/_fit_target_encoders/_try_lgbm/
_try_linear/_export_treg/_export_treg_blend)を直接呼び出して実際に学習し
(手書きバイナリ禁止、gen_cat_fixtures.py と同じ方針)、以下を生成する:

  - lgbm_datetime_parts_none_roundFalse   : datetime_parts(dt1: hour/dow/month/epoch_days)
                                            単独 + LightGBM(.treg v6)
  - blend_datetime_mixed_none_roundFalse  : one-hot(cat1) + target encoding(cat2) +
                                            datetime_parts(dt1) 混在のblend(.treg v6)

stress_test.csv に追加した dt1 列(区切りなし連結・T区切り・スラッシュ・日付のみ・秒省略・
空セル・範囲外値・閏年境界を含む)と組み合わせて run_matrix_test.js から検証する。

再現方法:
    python gen_datetime_fixtures.py
    (matrix/*.treg を書き込み、_manifest.json に統合する。stress_test.csv を使った
     matrix_cpp_out/*_pred.csv の再生成は README/predict-parity.yml と同じ手順で行うこと)
"""
import json
import os
import pickle
import sys
import tempfile

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _ROOT_DIR)

import train_bridge as tb  # noqa: E402

MATRIX_DIR = os.path.join(_THIS_DIR, "matrix")
MANIFEST_PATH = os.path.join(MATRIX_DIR, "_manifest.json")


def _mk_dir():
    return tempfile.mkdtemp(prefix="treg_gen_dt_")


def _write_and_copy(model_dir, out_name, expect_version):
    src = os.path.join(model_dir, "model.treg")
    dst = os.path.join(MATRIX_DIR, f"{out_name}.treg")
    with open(src, "rb") as f:
        data = f.read()
    with open(dst, "wb") as f:
        f.write(data)
    with open(dst, "rb") as f:
        ver = f.read(6)[4]
    print(f"  -> {dst} ({len(data)} bytes, v{ver})")
    assert ver == expect_version, f"expected file_version={expect_version}, got {ver}"


def _make_dt_strs(n, rng):
    """訓練用の日時文字列(複数フォーマット混在・区切りなし連結を含む)を生成する。
    真の信号(hour/dow/month)から y を構成し、モデルがdatetime_parts由来の4派生列を
    実際に使うことを検証する。"""
    base = pd.Timestamp("2016-01-01")
    offsets = rng.randint(0, 200 * 24 * 60, size=n)
    ts = [base + pd.Timedelta(minutes=int(o)) for o in offsets]

    def fmt(t, i):
        cyc = i % 5
        if cyc == 0:
            return t.strftime("%Y-%m-%d") + t.strftime("%H:%M:%S")  # 区切りなし連結
        if cyc == 1:
            return t.strftime("%Y-%m-%d %H:%M:%S")
        if cyc == 2:
            return t.strftime("%Y-%m-%dT%H:%M:%S")
        if cyc == 3:
            return t.strftime("%Y/%m/%d %H:%M:%S")
        return t.strftime("%Y-%m-%d %H:%M")  # 秒省略

    dt_strs = [fmt(t, i) for i, t in enumerate(ts)]
    hour = np.array([t.hour for t in ts], dtype=float)
    dow = np.array([t.dayofweek for t in ts], dtype=float)
    month = np.array([t.month for t in ts], dtype=float)
    return dt_strs, hour, dow, month


def gen_lgbm_datetime_parts():
    """1列の日時文字列(区切りあり/なし混在) → datetime_parts(4派生列) + LightGBM(.treg v6)。"""
    rng = np.random.RandomState(31)
    n = 300
    dt_strs, hour, dow, month = _make_dt_strs(n, rng)
    y = (np.sin(hour / 24 * 2 * np.pi) * 3.0 + dow * 0.5 + month * 0.2
         + rng.normal(0, 0.2, n))
    df = pd.DataFrame({"dt1": dt_strs, "y": y})

    df2, onehot_specs, target_cols, dropped, dt_specs = tb._prepare_categoricals(df, "y")
    assert not onehot_specs and not target_cols and not dropped, (onehot_specs, target_cols, dropped)
    assert len(dt_specs) == 4, dt_specs
    cat_encoders_all = dt_specs

    model_dir = _mk_dir()
    r2, feat_list, model_type, preds, info = tb._try_lgbm(
        df2, None, "y", model_dir, use_grid=False, use_oof=False,
        y_transform="none", y_params={}, df_all=None, num_jobs=1, splits=None)
    assert model_type == "lgbm", model_type

    ok = tb._export_treg("lgbm", model_dir, "y", y_transform="none", y_params={},
                         smear=1.0, y_clip=(-tb.X_CLIP_SENTINEL, tb.X_CLIP_SENTINEL),
                         round_output=False, x_clip_all={}, derived_recipe=[],
                         cat_encoders_all=cat_encoders_all)
    assert ok
    _write_and_copy(model_dir, "lgbm_datetime_parts_none_roundFalse", expect_version=6)
    return "lgbm_datetime_parts_none_roundFalse"


def gen_blend_datetime_mixed():
    """one-hot(cat1) + target encoding(cat2) + datetime_parts(dt1) 混在の2メンバーblend
    (linear=cat1+dt1中心、lgbm=cat2+dt1中心)。cat_encoders_all内でmethodが3種混在する
    ケース(file_version判定・writerのmethodディスパッチ)を検証する。"""
    rng = np.random.RandomState(32)
    n = 280
    x1 = rng.uniform(-3, 3, n)
    cat1 = rng.choice(["A", "B", "C"], size=n)
    classes2 = [f"R{i:02d}" for i in range(1, 13)]
    cat2 = rng.choice(classes2, size=n)
    dt_strs, hour, dow, month = _make_dt_strs(n, rng)
    effect1 = {"A": 1.0, "B": -2.0, "C": 4.0}
    effect2 = {c: (i - 6) * 1.2 for i, c in enumerate(classes2)}
    y = (1.2 * x1 + np.array([effect1[c] for c in cat1])
         + np.array([effect2[c] for c in cat2])
         + np.sin(hour / 24 * 2 * np.pi) * 2.0 + dow * 0.3
         + rng.normal(0, 0.3, n))
    df = pd.DataFrame({"x1": x1, "cat1": cat1, "cat2": cat2, "dt1": dt_strs, "y": y})

    df2, onehot_specs, target_cols, dropped, dt_specs = tb._prepare_categoricals(df, "y")
    assert not dropped, dropped
    assert len(dt_specs) == 4, dt_specs
    te_specs = tb._fit_target_encoders(df2, "y", target_cols)
    df3 = tb._apply_target_encoders(df2, te_specs)
    cat_encoders_all = onehot_specs + dt_specs + te_specs

    model_dir = _mk_dir()
    lin = tb._try_linear(df3, None, "y", model_dir, "none", {},
                         df_all=None, use_oof=False, splits=None)
    lgb = tb._try_lgbm(df3, None, "y", model_dir, use_grid=False, use_oof=False,
                       y_transform="none", y_params={}, df_all=None, num_jobs=1, splits=None)
    for name, res in (("Linear (Ridge)", lin), ("LightGBM", lgb)):
        assert res[0] is not None and np.isfinite(res[0]), f"{name} 学習失敗: {res}"

    candidates = {
        "Linear (Ridge)": (lin[0], lin[1], "linear", None, {}),
        "LightGBM":       (lgb[0], lgb[1], "lgbm",   None, {}),
    }
    weights = {"Linear (Ridge)": 0.5, "LightGBM": 0.5}
    with open(os.path.join(model_dir, "blend_meta.pkl"), "wb") as f:
        pickle.dump({"models": list(candidates.keys()), "weights": weights}, f)

    ok = tb._export_treg_blend(model_dir, "y", candidates, y_transform="none", y_params={},
                               smear=1.0, y_clip=(-tb.X_CLIP_SENTINEL, tb.X_CLIP_SENTINEL),
                               round_output=False, x_clip_all={}, derived_recipe=[],
                               cat_encoders_all=cat_encoders_all)
    assert ok, "blend export failed"
    # blendの外側ラッパー自身は feat_cols=[] (blendは直接特徴を持たない、既存のv4/v5と同じ挙動)
    # のため used_cat が常に空になり、外側の file_version はv3のまま(datetimeが混在していても
    # 上がらない)。各メンバー(linear/lgbm)がそれぞれ自己完結した入れ子.tregとしてv6を持つ。
    _write_and_copy(model_dir, "blend_datetime_mixed_none_roundFalse", expect_version=3)
    return "blend_datetime_mixed_none_roundFalse"


def main():
    os.makedirs(MATRIX_DIR, exist_ok=True)
    new_names = []
    new_names.append(gen_lgbm_datetime_parts())
    new_names.append(gen_blend_datetime_mixed())

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for name in new_names:
        if name not in manifest:
            manifest.append(name)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\n_manifest.json に {len(new_names)} 件追加(合計 {len(manifest)} 件)")
    print("次に stress_test.csv(dt1列追加済み)で予測を実行し、"
          "C++参照実装(predict_native_ref)の出力を matrix_cpp_out/ に生成すること。")


if __name__ == "__main__":
    main()
