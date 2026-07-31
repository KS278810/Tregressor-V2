"""_ag_benchmark/gen_real.py

実データの回帰問題を 6 問取得する(合成24 + 実6 = 30問)。AG venv(sklearn入り)で実行:
    agenv/Scripts/python.exe _ag_benchmark/gen_real.py

sklearn 同梱(diabetes / california)+ OpenML(fetch_openml)から取得。
- 大きいデータは MAX_ROWS=2000 に固定シードでサブサンプル(AGとtregの学習時間を現実的に保つ)
- target を数値強制できないもの(真の分類)はスキップ
- 実データは ceiling_r2 不明 → null

候補を優先順で試し、成功した先頭6件を採用。ネットワーク不通で一部落ちても続行する。
出力: data/<name>.csv + data/manifest_real.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data")
MAX_ROWS = 2000
SUBSAMPLE_SEED = 7
N_TARGET = 6


def _subsample(df):
    if len(df) > MAX_ROWS:
        idx = np.random.RandomState(SUBSAMPLE_SEED).permutation(len(df))[:MAX_ROWS]
        df = df.iloc[idx].reset_index(drop=True)
    return df


def _finalize(X, y, name, note):
    """X(DataFrame), y(Series) を1つのCSV仕様に整える。target 列名は 'target'。"""
    y = pd.to_numeric(y, errors="coerce")
    df = X.copy()
    df["target"] = y.values
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    if df["target"].nunique() < 5:
        raise ValueError(f"target のユニーク値が少なすぎる({df['target'].nunique()}) → 分類の疑い、スキップ")
    df = _subsample(df)
    return df, "target", note


# 各ローダは (df, target_col, note) を返す。例外は呼び出し側で捕捉しスキップ。
def load_diabetes_():
    from sklearn.datasets import load_diabetes
    d = load_diabetes(as_frame=True)
    return _finalize(d.data, d.target, "real_diabetes", "sklearn同梱。糖尿病進行度(n=442,10特徴)")


def load_california_():
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing(as_frame=True)
    return _finalize(d.data, d.target, "real_california", "住宅価格中央値(2万→2000にサブサンプル)")


def _openml(name_or_id, note, tag, version="active"):
    from sklearn.datasets import fetch_openml
    kw = dict(as_frame=True, parser="auto")
    if isinstance(name_or_id, int):
        d = fetch_openml(data_id=name_or_id, **kw)
    else:
        d = fetch_openml(name=name_or_id, version=version, **kw)
    X = d.data
    # 全欠損/定数列は落とす。カテゴリ文字列はそのまま(両ツールが処理する)
    X = X.dropna(axis=1, how="all")
    return _finalize(X, d.target, tag, note)


def load_concrete_():   return _openml(4353, "コンクリート圧縮強度(n=1030)", "real_concrete")
def load_energy_():     return _openml(43439, "建物エネルギー効率(冷暖房負荷)", "real_energy")
def load_abalone_():    return _openml(183, "アワビの年齢(輪の数, n=4177→2000)", "real_abalone")
def load_wine_():       return _openml(287, "赤ワイン品質スコア回帰(n=1599)", "real_winequality")
def load_airfoil_():    return _openml(44957, "翼型自己雑音の音圧レベル(n=1503)", "real_airfoil")
def load_cpu_():        return _openml(227, "CPU相対性能(cpu_act系)", "real_cpu")


CANDIDATES = [
    load_diabetes_, load_california_, load_concrete_, load_energy_,
    load_abalone_, load_wine_, load_airfoil_, load_cpu_,
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {}
    for loader in CANDIDATES:
        if len(manifest) >= N_TARGET:
            break
        try:
            df, target, note = loader()
        except Exception as e:
            print(f"  スキップ {loader.__name__}: {str(e)[:150]}")
            continue
        # tag は _finalize には渡していないので loader 名から復元
        name = {
            "load_diabetes_": "real_diabetes", "load_california_": "real_california",
            "load_concrete_": "real_concrete", "load_energy_": "real_energy",
            "load_abalone_": "real_abalone", "load_wine_": "real_winequality",
            "load_airfoil_": "real_airfoil", "load_cpu_": "real_cpu",
        }[loader.__name__]
        csv_name = f"{name}.csv"
        df.to_csv(os.path.join(OUT_DIR, csv_name), index=False, encoding="utf-8")
        manifest[name] = {
            "csv": csv_name, "target": target, "n": int(len(df)),
            "n_cols": int(df.shape[1] - 1), "family": "real",
            "ceiling_r2": None, "source": "real", "note": note,
        }
        print(f"取得: {name:20s} n={len(df):5d} cols={df.shape[1]-1:3d}  {note}")

    with open(os.path.join(OUT_DIR, "manifest_real.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"\n完了: 実データ {len(manifest)} 問。")
    if len(manifest) < N_TARGET:
        print(f"[警告] 目標 {N_TARGET} に届かず({len(manifest)}件)。合計は 24+{len(manifest)} 問。", file=sys.stderr)


if __name__ == "__main__":
    main()
