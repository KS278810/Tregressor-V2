"""tests/benchmarks/gen_datasets.py

リリース前ベンチマークスイート(tests/run_benchmark.py)用の合成データセット生成器。

決定論的(全データセットが固定シード)に、T-regressorの主要な挙動分岐
(FE/カテゴリエンコーダのone-hot・target encoding・高カーディナリティ除外/欠損補完/
外れ値クリップ/歪みy変換(log1p・yeo-johnson)/小データ/重複行/文字エンコーディング境界/
リーク検知)を踏むデータを300〜1000行(small_nのみ30行)で生成する。

実行方法(システムPython。numpy/pandasのみに依存、sklearn/scipy不要):
    python tests/benchmarks/gen_datasets.py

出力先: tests/benchmarks/data/*.csv + manifest.json。生成物はgit管理下に置く
("実行毎に再生成しても同一バイト列になる"ことがベンチマークのbaseline比較の前提のため。
再現性を壊す変更をした場合はこのスクリプトを実行し直してdata/を差分コミットすること)。
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data")


def _rng(seed):
    return np.random.RandomState(seed)


def linear_clean(n=600, seed=101):
    """きれいな線形関係。quick(linear)で高R²が出るはずの基準ケース。"""
    rng = _rng(seed)
    X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(1, 6)})
    y = 3.0 * X["x1"] - 2.0 * X["x2"] + 0.5 * X["x3"] + rng.randn(n) * 0.2
    X["y"] = y
    return X, "y"


def nonlinear_interaction(n=700, seed=102):
    """非線形+交互作用。thoroughの自動FE/blendがquickより明確に勝つはずのケース。"""
    rng = _rng(seed)
    X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(1, 7)})
    y = (np.sin(X["x1"] * 2) * 3 + X["x2"] * X["x3"] + np.where(X["x4"] > 0, 2, -2)
         + 0.4 * X["x5"] ** 2 + rng.randn(n) * 1.2)
    X["y"] = y
    return X, "y"


def categorical_low(n=500, seed=103):
    """低カーディナリティ(5、CAT_ONEHOT_MAX_CARD=10以下)カテゴリ列。one-hot経路。"""
    rng = _rng(seed)
    classes = list("ABCDE")
    effects = _rng(seed + 1).randn(len(classes)) * 4
    idx = rng.randint(0, len(classes), size=n)
    grade = np.array(classes)[idx]
    x1 = rng.randn(n)
    y = 1.5 * x1 + effects[idx] + rng.randn(n) * 0.5
    return pd.DataFrame({"x1": x1, "grade": grade, "y": y}), "y"


def categorical_high(n=500, seed=104):
    """高カーディナリティカテゴリ列。target encoding経路(10<card<=行数の50%、card=30)と、
    カーディナリティが行数の50%を超える列除外(cat_dropped_columns、card=n)の両方を踏む。"""
    rng = _rng(seed)
    n_city = 30
    cities = [f"city_{i}" for i in range(n_city)]
    effects = _rng(seed + 1).randn(n_city) * 4
    idx = rng.randint(0, n_city, size=n)
    city = np.array(cities)[idx]
    x1 = rng.randn(n)
    raw_id = np.array([f"id_{i}" for i in range(n)])  # card==n → 除外対象
    y = x1 + effects[idx] + rng.randn(n) * 0.5
    return pd.DataFrame({"x1": x1, "city": city, "raw_id": raw_id, "y": y}), "y"


def bool_mixed(n=400, seed=105):
    """bool型(True/False)列と数値の混在。"""
    rng = _rng(seed)
    flag = rng.rand(n) < 0.5
    x1 = rng.randn(n)
    y = x1 * 2 + np.where(flag, 3.0, -3.0) + rng.randn(n) * 0.4
    return pd.DataFrame({"x1": x1, "flag": flag, "y": y}), "y"


def missing_heavy(n=500, seed=106):
    """欠損が重い(列ごと20〜35%)。median補完・欠損データ警告を踏む。"""
    rng = _rng(seed)
    X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(1, 5)})
    y = X["x1"] + X["x2"] * 0.5 + rng.randn(n) * 0.3
    X["y"] = y
    for i, col in enumerate(["x1", "x2", "x3", "x4"]):
        frac = 0.20 + 0.05 * i
        mask = rng.rand(n) < frac
        X.loc[mask, col] = np.nan
    return X, "y"


def skewed_target(n=500, seed=107):
    """歪んだy(対数正規)。log1p/yeo-johnson変換分岐を踏む。"""
    rng = _rng(seed)
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    y = np.exp(1.0 + 0.7 * x1 + rng.randn(n) * 0.3)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y}), "y"


def outlier_contaminated(n=500, seed=108):
    """外れ値汚染(2%の行に巨大な外れ値)。IQRクリップに基づくdata_warningを踏む。"""
    rng = _rng(seed)
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    y = 2 * x1 + rng.randn(n) * 0.5
    n_out = max(3, int(n * 0.02))
    out_idx = rng.choice(n, size=n_out, replace=False)
    y = y.copy()
    y[out_idx] = y[out_idx] + rng.choice([1.0, -1.0], size=n_out) * rng.uniform(30, 50, size=n_out)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y}), "y"


def small_n(n=30, seed=109):
    """小データ(n=30)。CV/自動FEスキップの境界(MIN_ROWS_FOR_SPLIT=10)を余裕を持って踏む。"""
    rng = _rng(seed)
    X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(1, 4)})
    y = X["x1"] - X["x2"] + rng.randn(n) * 0.3
    X["y"] = y
    return X, "y"


def pure_noise(n=500, seed=110):
    """リーク検知番犬。XとyはRandomState完全独立の無関係な乱数。
    パイプラインにリーク(例: fold外fitのtarget encoding等)が無ければOOF R²は0近傍/負のはず。
    R²>0.15はリーク混入の疑いが強いという異常検知の基準データ(run_benchmark.py側で判定)。"""
    rng_x = _rng(seed)
    rng_y = _rng(seed + 1000)  # Xと完全独立の別RandomState
    X = pd.DataFrame({f"x{i}": rng_x.randn(n) for i in range(1, 9)})
    X["y"] = rng_y.randn(n)
    return X, "y"


def duplicated_rows(n=300, seed=111):
    """重複行(元n行をそのまま2回連結)。重複行検出のdata_warningを踏む。"""
    rng = _rng(seed)
    X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(1, 5)})
    y = X["x1"] * 1.5 - X["x2"] + rng.randn(n) * 0.3
    X["y"] = y
    return pd.concat([X, X.copy()], ignore_index=True), "y"


def jp_columns(n=400, seed=112):
    """日本語列名+日本語カテゴリ値。呼び出し側(main)がUTF-8版とShift-JIS(cp932)版の
    2ファイルとして書き出す(_read_csv_with_encoding_fallbackのUTF-8/cp932判定分岐を踏む)。"""
    rng = _rng(seed)
    kion = rng.randn(n) * 5 + 20
    shitsudo = rng.randn(n) * 10 + 50
    areas = ["東京", "大阪", "札幌"]
    area_effects = _rng(seed + 1).randn(len(areas))
    idx = rng.randint(0, len(areas), size=n)
    chiiki = np.array(areas)[idx]
    uriage = kion * 0.3 - shitsudo * 0.1 + area_effects[idx] * 10 + rng.randn(n) * 2
    df = pd.DataFrame({"温度": kion, "湿度": shitsudo, "地域": chiiki, "売上高": uriage})
    return df, "売上高"


# (関数, target_colは各関数の戻り値2番目) の順。dict順=マニフェスト内の記載順。
GENERATORS = {
    "linear_clean": linear_clean,
    "nonlinear_interaction": nonlinear_interaction,
    "categorical_low": categorical_low,
    "categorical_high": categorical_high,
    "bool_mixed": bool_mixed,
    "missing_heavy": missing_heavy,
    "skewed_target": skewed_target,
    "outlier_contaminated": outlier_contaminated,
    "small_n": small_n,
    "pure_noise": pure_noise,
    "duplicated_rows": duplicated_rows,
    "jp_columns": jp_columns,
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {}
    for name, fn in GENERATORS.items():
        df, target = fn()
        if name == "jp_columns":
            utf8_path = os.path.join(OUT_DIR, "jp_columns.csv")
            df.to_csv(utf8_path, index=False, encoding="utf-8")
            manifest["jp_columns"] = {"csv": "jp_columns.csv", "target": target,
                                       "note": "UTF-8。日本語列名+日本語カテゴリ値。"}
            sjis_path = os.path.join(OUT_DIR, "jp_columns_sjis.csv")
            df.to_csv(sjis_path, index=False, encoding="cp932")
            manifest["jp_columns_sjis"] = {"csv": "jp_columns_sjis.csv", "target": target,
                                            "note": "Shift-JIS(cp932)。内容はjp_columnsと同一、"
                                                    "エンコーディング判定分岐の確認用。"}
        else:
            csv_name = f"{name}.csv"
            df.to_csv(os.path.join(OUT_DIR, csv_name), index=False, encoding="utf-8")
            manifest[name] = {"csv": csv_name, "target": target, "note": f"n={len(df)}"}
        print(f"生成: {name} (n={len(df)}, target={target!r})")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"\n完了: {len(manifest)}件のデータセットを {OUT_DIR} に生成しました。")


if __name__ == "__main__":
    main()
