"""gen_composite_key_fixtures.py — .treg v7 composite_target カテゴリエンコーダ
パリティフィクスチャ生成(2026-07第3弾・真因②対策「反復測定データの被験者代理キーが
未活用」対策)。

train_bridge.py の内部関数(_detect_numeric_composite_key/_fit_target_encoders/
_try_lgbm/_try_linear/_export_treg/_export_treg_blend)を直接呼び出して実際に学習し
(手書きバイナリ禁止、gen_cat_fixtures.py/gen_datetime_fixtures.py と同じ方針)、以下を
生成する:

  - lgbm_composite_key_none_roundFalse   : 合成キー(grp_a+grp_b)単独 + LightGBM(.treg v7)
  - blend_composite_key_mixed_none_roundFalse : one-hot(cat1) + 合成キー(grp_a+grp_b)
                                                混在のblend(.treg v7)

stress_test.csv に追加した grp_a/grp_b 列(低カーディナリティ数値列、欠損セルを含む)と
組み合わせて run_matrix_test.js から検証する。

再現方法:
    python gen_composite_key_fixtures.py
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
    return tempfile.mkdtemp(prefix="treg_gen_ckey_")


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


def _make_grouped_data(n_groups, rows_per_group, rng):
    """反復測定構造(グループ=擬似被験者)の合成データ。grp_a/grp_bの組み合わせが
    実質的なグループIDになる(stress_test.csvのgrp_a/grp_bと同じ形)。"""
    grp_a_vals = rng.choice(np.arange(0, 10), n_groups, replace=False if n_groups <= 10 else True)
    grp_b_vals = rng.randint(0, 3, n_groups)
    grp_effect = rng.normal(0, 5, n_groups)

    grp_a, grp_b, x1, grp_idx = [], [], [], []
    for g in range(n_groups):
        for _ in range(rows_per_group):
            grp_a.append(float(grp_a_vals[g]))
            grp_b.append(float(grp_b_vals[g]))
            x1.append(rng.normal(0, 1))
            grp_idx.append(g)
    grp_a = np.array(grp_a)
    grp_b = np.array(grp_b)
    x1 = np.array(x1)
    grp_effect_per_row = np.array([grp_effect[g] for g in grp_idx])
    y = 0.2 * x1 + grp_effect_per_row + rng.normal(0, 0.3, len(x1))
    return grp_a, grp_b, x1, y


def gen_lgbm_composite_key():
    """grp_a+grp_bの組み合わせ(合成キー) + LightGBM(.treg v7)。"""
    rng = np.random.RandomState(41)
    grp_a, grp_b, x1, y = _make_grouped_data(n_groups=15, rows_per_group=20, rng=rng)
    df = pd.DataFrame({"grp_a": grp_a, "grp_b": grp_b, "x1": x1, "y": y})

    df2, keycol, srccols = tb._detect_numeric_composite_key(df, "y")
    assert keycol == "__numkey__grp_a_grp_b", keycol
    assert srccols == ["grp_a", "grp_b"], srccols

    te_specs = tb._fit_target_encoders(df2, "y", [keycol])
    assert len(te_specs) == 1, te_specs
    df3 = tb._apply_target_encoders(df2, te_specs)
    te_specs[0]["method"] = "composite_target"
    te_specs[0]["source_col"] = tb.NUMERIC_KEY_SEP.join(srccols)
    cat_encoders_all = te_specs

    model_dir = _mk_dir()
    r2, feat_list, model_type, preds, info = tb._try_lgbm(
        df3, None, "y", model_dir, use_grid=False, use_oof=False,
        y_transform="none", y_params={}, df_all=None, num_jobs=1, splits=None)
    assert model_type == "lgbm", model_type

    ok = tb._export_treg("lgbm", model_dir, "y", y_transform="none", y_params={},
                         smear=1.0, y_clip=(-tb.X_CLIP_SENTINEL, tb.X_CLIP_SENTINEL),
                         round_output=False, x_clip_all={}, derived_recipe=[],
                         cat_encoders_all=cat_encoders_all)
    assert ok
    _write_and_copy(model_dir, "lgbm_composite_key_none_roundFalse", expect_version=7)
    return "lgbm_composite_key_none_roundFalse"


def gen_blend_composite_key_mixed():
    """one-hot(cat1) + 合成キー(grp_a+grp_b) 混在の2メンバーblend
    (linear=cat1中心、lgbm=合成キー中心)。"""
    rng = np.random.RandomState(42)
    grp_a, grp_b, x1, y_grp = _make_grouped_data(n_groups=15, rows_per_group=18, rng=rng)
    n = len(x1)
    cat1 = rng.choice(["A", "B", "C"], size=n)
    effect1 = {"A": 1.0, "B": -2.0, "C": 4.0}
    y = y_grp + np.array([effect1[c] for c in cat1])
    df = pd.DataFrame({"grp_a": grp_a, "grp_b": grp_b, "x1": x1, "cat1": cat1, "y": y})

    df2, onehot_specs, target_cols, dropped, dt_specs = tb._prepare_categoricals(df, "y")
    assert not dropped and not dt_specs, (dropped, dt_specs)
    exclude = {s["feature_name"] for s in onehot_specs} | {s["feature_name"] for s in dt_specs}
    df3, keycol, srccols = tb._detect_numeric_composite_key(df2, "y", exclude_cols=exclude)
    assert keycol == "__numkey__grp_a_grp_b", keycol

    te_specs = tb._fit_target_encoders(df3, "y", target_cols + [keycol])
    df4 = tb._apply_target_encoders(df3, te_specs)
    for spec in te_specs:
        if spec["feature_name"] == keycol:
            spec["method"] = "composite_target"
            spec["source_col"] = tb.NUMERIC_KEY_SEP.join(srccols)
    cat_encoders_all = onehot_specs + te_specs

    model_dir = _mk_dir()
    lin = tb._try_linear(df4, None, "y", model_dir, "none", {},
                         df_all=None, use_oof=False, splits=None)
    lgb = tb._try_lgbm(df4, None, "y", model_dir, use_grid=False, use_oof=False,
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
    # 外側のblendラッパー自身はfeat_cols=[]のためused_catが常に空になりv3のまま
    # (datetime/onehot/target同様の既知の挙動)。各メンバーがそれぞれ入れ子.tregとして
    # v7を持つ。
    _write_and_copy(model_dir, "blend_composite_key_mixed_none_roundFalse", expect_version=3)
    return "blend_composite_key_mixed_none_roundFalse"


def main():
    os.makedirs(MATRIX_DIR, exist_ok=True)
    new_names = []
    new_names.append(gen_lgbm_composite_key())
    new_names.append(gen_blend_composite_key_mixed())

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for name in new_names:
        if name not in manifest:
            manifest.append(name)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\n_manifest.json に {len(new_names)} 件追加(合計 {len(manifest)} 件)")
    print("次に stress_test.csv(grp_a/grp_b列追加済み)で予測を実行し、"
          "C++参照実装(predict_native_ref)の出力を matrix_cpp_out/ に生成すること。")


if __name__ == "__main__":
    main()
