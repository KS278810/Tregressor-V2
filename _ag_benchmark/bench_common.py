"""_ag_benchmark/bench_common.py

T-regressor vs AutoGluon 公平比較ハーネスの共通部品。
- 決定論的な train/test 分割(両ツールで同一分割を保証するため、分割結果はファイルに書き出し、
  AG 側はそれを読むだけにする)
- 中立採点器(純 numpy。両ツールの予測を同一コードで採点することが公平性の核心)
numpy/pandas のみに依存(embed python でも AG venv でも動く)。
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # リポジトリルート
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")
WORK_DIR = os.path.join(RESULTS_DIR, "_work")

SPLIT_SEED = 42
TEST_FRAC = 0.25
MIN_TEST = 8


def read_any_csv(path):
    """UTF-8(BOM有無)→cp932 の順でフォールバックして読む。"""
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)  # 最後の手段(既定エンコーディング)


def make_split(df, target, seed=SPLIT_SEED, test_frac=TEST_FRAC):
    """決定論的シャッフル分割。(train_df, test_df) を返す。index はリセットする。"""
    n = len(df)
    perm = np.random.RandomState(seed).permutation(n)
    n_test = max(MIN_TEST, int(round(n * test_frac)))
    n_test = min(n_test, n - MIN_TEST)  # train も最低限確保
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


def write_split(name, df, target):
    """<name> の train/test/test_features を _work/<name>/ に書き出し、パス辞書を返す。
    両ツールがこの同一ファイルを使うことで分割の同一性を保証する。"""
    wd = os.path.join(WORK_DIR, name)
    os.makedirs(wd, exist_ok=True)
    train_df, test_df = make_split(df, target)
    test_feat = test_df.drop(columns=[target])
    paths = {
        "train": os.path.join(wd, "train.csv"),
        "test": os.path.join(wd, "test.csv"),
        "test_features": os.path.join(wd, "test_features.csv"),
        "workdir": wd,
    }
    train_df.to_csv(paths["train"], index=False, encoding="utf-8")
    test_df.to_csv(paths["test"], index=False, encoding="utf-8")
    test_feat.to_csv(paths["test_features"], index=False, encoding="utf-8")
    return paths, len(train_df), len(test_df)


def score(y_true, y_pred):
    """中立採点器。(r2, rmse, mae, n_used) を返す。非有限値はペア単位で除外。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return None, None, None, 0
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    return round(r2, 4), round(rmse, 4), round(mae, 4), int(len(yt))


def load_manifest(*names):
    """複数の manifest_*.json をマージして返す(synth と real)。"""
    merged = {}
    for nm in names:
        p = os.path.join(DATA_DIR, nm)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                merged.update(json.load(f))
    return merged


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path):
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs
