"""_ag_benchmark/gen_public.py

第二弾: OpenML-CTR23 / AutoML-Benchmark(AMLB)が定義する「表形式回帰の学術標準ベンチマーク」の
選定哲学(適切なサイズ・非スパース・i.i.d.・出典明記)に沿う、UCI ML Repository 由来の
著名な公開回帰データセットを追加取得する。

背景: OpenML の study/data API が本作業時点で応答不能(504多発、8秒タイムアウトでも無応答)
だったため、CTR23 suite(id=353)のメンバーシップそのものは実行時に検証できなかった。
そのため「CTR23公式リストの完全再現」ではなく、CTR23/AMLBが典拠として重視するのと同じ
UCI ML Repository から、ML文献で広く参照される著名データセットを個別に選定し、
ucimlrepo経由(OpenMLと独立したUCI公式インフラ)で直接取得する。事実確認: 各データセットの
UCI公式IDとページは https://archive.ics.uci.edu/dataset/<id> で人手検証可能。

既存の30問(合成24+実6)には触れない。追加分は data/manifest_public.json に分離する。
実行: agenv/Scripts/python.exe _ag_benchmark/gen_public.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data")
MAX_ROWS = 2000
SUBSAMPLE_SEED = 11

# (uci_id, tag, target_col_override(Noneならy.columns[0]), 削除する追加ターゲット/ID列, note)
CANDIDATES = [
    (165, "pub_concrete",   None, [],
     "コンクリート圧縮強度(UCI 165, n=1030)。CTR23/AMLB系論文で頻出の定番"),
    (242, "pub_energy",     "Heating_Load", ["Cooling_Load"],
     "建物エネルギー効率(UCI 242, n=768)。暖房負荷を予測、冷房負荷列は除外"),
    (294, "pub_powerplant", None, [],
     "複合サイクル発電所の出力(UCI 294, n=9568→2000)。4特徴のみの低次元回帰"),
    (9,   "pub_automgp",    None, [],
     "自動車燃費 Auto MPG(UCI 9, n=398)。カテゴリ列(原産地)混在の古典データ"),
    (477, "pub_realestate", None, [],
     "台湾不動産価格(UCI 477, n=414)。小標本の実務データ"),
    (243, "pub_yacht",      None, [],
     "ヨット船体の造波抵抗(UCI 243, n=308)。低次元・小標本"),
    (162, "pub_forestfires", None, [],
     "森林火災の焼失面積(UCI 162, n=517)。極端な右裾(0が大半)、頑健性の試金石"),
    (189, "pub_parkinsons", "total_UPDRS", ["motor_UPDRS"],
     "パーキンソン病重症度スコア(UCI 189, n=5875→2000)。total_UPDRSを予測"),
    (464, "pub_superconduct", None, [],
     "超伝導臨界温度(UCI 464, n=21263→2000, 81特徴)。高次元・表形式DL論文の定番"),
    (504, "pub_fishtoxicity", None, [],
     "QSAR魚毒性予測(UCI 504, n=908)。化学記述子からの回帰"),
    (374, "pub_appliances",  None, [],
     "家電エネルギー消費予測(UCI 374, n=19735→2000, 高次元時系列由来特徴)"),
    (320, "pub_studentperf", "G3", ["G1", "G2"],
     "生徒の最終成績(UCI 320, n=395)。G1/G2(中間成績)は最終成績と強相関のため除外し純粋な特徴から予測"),
]


def _subsample(df):
    if len(df) > MAX_ROWS:
        idx = np.random.RandomState(SUBSAMPLE_SEED).permutation(len(df))[:MAX_ROWS]
        df = df.iloc[idx].reset_index(drop=True)
    return df


def fetch_one(uci_id, tag, target_override, drop_extra, note):
    repo = fetch_ucirepo(id=uci_id)
    X = repo.data.features.copy()
    Y = repo.data.targets.copy()
    if target_override is not None and target_override in Y.columns:
        y = Y[target_override]
    else:
        y = Y.iloc[:, 0]
    target_name = "target"
    for col in drop_extra:
        if col in X.columns:
            X = X.drop(columns=[col])
    df = X.copy()
    df[target_name] = pd.to_numeric(y, errors="coerce").values
    df = df.dropna(subset=[target_name]).reset_index(drop=True)
    df = df.dropna(axis=1, how="all")
    if df[target_name].nunique() < 5:
        raise ValueError(f"target のユニーク値が少なすぎる({df[target_name].nunique()})")
    df = _subsample(df)
    return df, target_name


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {}
    for uci_id, tag, target_override, drop_extra, note in CANDIDATES:
        try:
            df, target = fetch_one(uci_id, tag, target_override, drop_extra, note)
        except Exception as e:
            print(f"  スキップ {tag}(uci_id={uci_id}): {str(e)[:150]}")
            continue
        csv_name = f"{tag}.csv"
        df.to_csv(os.path.join(OUT_DIR, csv_name), index=False, encoding="utf-8")
        manifest[tag] = {
            "csv": csv_name, "target": target, "n": int(len(df)),
            "n_cols": int(df.shape[1] - 1), "family": "public",
            "ceiling_r2": None, "source": "public_uci", "uci_id": uci_id, "note": note,
        }
        print(f"取得: {tag:20s} n={len(df):5d} cols={df.shape[1]-1:3d}  (UCI id={uci_id})  {note}")

    with open(os.path.join(OUT_DIR, "manifest_public.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"\n完了: 公開ベンチマーク {len(manifest)} 問 / 候補 {len(CANDIDATES)} 問。")


if __name__ == "__main__":
    main()
