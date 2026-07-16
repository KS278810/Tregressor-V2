#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/test_corrupt_treg.py

重大-1(.tregリーダーの整合性検査)・中-M1(blend再帰深さ制限)の回帰テスト。

predict_native_v2.cpp の load_treg は、破損した/悪意を持って細工された .treg
バイナリを読んでもヒープ範囲外読み出し(UB)・無限ループ・クラッシュを起こさず、
必ず「読み込み失敗」としてエラー終了する(exit code 0 or 1)ことが要件になっている。
このスクリプトは既存の正常な .treg フィクスチャから

  (a) 末尾を数バイト削ったバージョン(truncate)
  (b) 末尾を伸ばしてゴミバイトを追記したバージョン(extend)
  (c) 4バイト境界の随所を 0xFFFFFFFF で上書きしたバージョン
      (n_fc・n_feat・n_trees・n_leaves 等のいずれかのカウントフィールドを
       確実に踏む「n_fc改竄」に相当する)

を大量に生成し、native 推論器(predict_native_v2.cpp をビルドしたexe)に
実際に読み込ませて、クラッシュ相当の異常終了(シグナル・Windowsのアクセス違反
0xC0000005・タイムアウト)が一切発生しないことを確認する。

使い方:
  python tests/test_corrupt_treg.py [native_exe_path]
  (省略時は native_predictor/predict_native.exe → predict_native_ref(.exe) の
   順にリポジトリ内を探索する。ビルドされていなければ [SKIP] して exit 0 で終わる
   — CIの他ジョブを壊さないため)

事前準備(未ビルドの場合):
  g++ -O2 -std=c++17 native_predictor/predict_native_v2.cpp -o native_predictor/predict_native_ref
  (Windows/MinGWの場合は -std=gnu++17 -lshell32 を追加。native_predictor/build_native.ps1参照)

各バリアントには環境変数 TREG_NO_GUI=1 を設定して起動する。これは fatal() が
Windowsで MessageBoxW によるモーダルダイアログを出す既定動作を抑止するための
テスト専用フック(predict_native_v2.cpp 参照)で、これが無いと「読み込み失敗」時に
自動テストがクリックする人間を待って無期限にハングしてしまう。
"""
import os
import random
import struct
import subprocess
import sys
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (壊す元になる正常な.treg, 対応するCSV, 4バイト境界の走査間隔)。
# linear(単純な固定長ペイロード)・lgbm(木構造、重大-1のノード範囲検査の主対象)・
# blend(入れ子.treg、中-M1の再帰深さ制限の主対象)の3種で異なるコードパスを踏む。
# step=1にしてある2つ(linear/blend)はファイルが小さく、全バイトオフセットを
# 走査してもn_fc等のカウントフィールドを確実に踏める(取りこぼしなし)。
FIXTURES = [
    ("web/js_predict_poc/sample_linear_model.treg", "web/js_predict_poc/sample_strict.csv", 1),
    ("web/js_predict_poc/sample_lgbm_model_noround.treg", "web/js_predict_poc/sample_strict.csv", 32),
    ("web/js_predict_poc/matrix/blend_lgbm_linear_log1p_roundFalse.treg", "web/js_predict_poc/stress_test.csv", 1),
]
MAX_TAMPER_PER_FIXTURE = 400  # 1フィクスチャあたりの改竄バリアント数上限(実行時間対策)


def find_native_exe(explicit):
    candidates = []
    if explicit:
        # subprocess.run 呼び出し時に cwd を一時ディレクトリへ切り替えるため、相対パスの
        # ままだと解決先がずれてFileNotFoundErrorになる。ここで絶対パス化しておく。
        candidates.append(os.path.abspath(explicit))
    candidates += [
        os.path.join(ROOT, "native_predictor", "predict_native.exe"),
        os.path.join(ROOT, "native_predictor", "predict_native_ref.exe"),
        os.path.join(ROOT, "native_predictor", "predict_native_ref"),
        os.path.join(ROOT, "web", "js_predict_poc", "predict_native_ref.exe"),
        os.path.join(ROOT, "web", "js_predict_poc", "predict_native_ref"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def gen_variants(data: bytes, tamper_step: int):
    """(バリアント名, バイト列) の辞書を生成する。"""
    variants = {}
    n = len(data)

    # (a) 末尾を数バイト削る
    for cut in (1, 2, 4, 8, 16, 64, max(1, n // 2)):
        if 0 < cut < n:
            variants[f"truncate_last_{cut}"] = data[: n - cut]
    variants["truncate_to_5bytes"] = data[: min(5, n)]
    variants["truncate_to_0bytes"] = b""

    # (b) 末尾を伸ばす(ゴミバイト追記)
    rnd = random.Random(12345)
    for extra in (1, 16, 1024):
        variants[f"extend_{extra}"] = data + bytes(rnd.randrange(256) for _ in range(extra))

    # (c) 4バイト境界ごとに 0xFFFFFFFF を書き込む(n_fc等のカウントフィールド改竄)
    offsets = list(range(0, max(0, n - 4), tamper_step))
    if len(offsets) > MAX_TAMPER_PER_FIXTURE:
        step2 = len(offsets) // MAX_TAMPER_PER_FIXTURE + 1
        offsets = offsets[::step2]
    for off in offsets:
        mutated = bytearray(data)
        struct.pack_into("<I", mutated, off, 0xFFFFFFFF)
        variants[f"tamper_u32_at_{off}"] = bytes(mutated)

    return variants


def looks_like_crash(returncode, timed_out):
    if timed_out:
        return True
    # 0=正常終了、1=fatal()経由の明示エラー終了(load_treg失敗・LGBM巡回上限超過の
    # runtime_errorなど、いずれも本実装ではエラー終了として正しい)。
    # それ以外(POSIXシグナルによる負のreturncode、Windowsのアクセス違反
    # 0xC0000005 → Pythonでは -1073741819 等)はクラッシュとみなす。
    return returncode not in (0, 1)


def main():
    explicit = sys.argv[1] if len(sys.argv) > 1 else None
    exe = find_native_exe(explicit)
    if not exe:
        print("[SKIP] native実装のexeが見つかりません。以下でビルドしてから再実行してください:")
        print("  g++ -O2 -std=c++17 native_predictor/predict_native_v2.cpp -o native_predictor/predict_native_ref")
        return 0

    print(f"native exe: {exe}")
    env = dict(os.environ)
    env["TREG_NO_GUI"] = "1"

    total = 0
    crashes = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for treg_rel, csv_rel, tamper_step in FIXTURES:
            treg_path = os.path.join(ROOT, treg_rel)
            csv_path = os.path.join(ROOT, csv_rel)
            if not os.path.isfile(treg_path) or not os.path.isfile(csv_path):
                print(f"[SKIP] フィクスチャ不足: {treg_rel} / {csv_rel}")
                continue
            with open(treg_path, "rb") as f:
                base = f.read()
            variants = gen_variants(base, tamper_step)
            print(f"\n=== {os.path.basename(treg_rel)}: {len(variants)}バリアント ===")
            for name, blob in variants.items():
                total += 1
                treg_out = os.path.join(tmpdir, f"{name}.treg")
                csv_out = os.path.join(tmpdir, f"{name}.csv")
                with open(treg_out, "wb") as f:
                    f.write(blob)
                with open(csv_path, "rb") as fsrc, open(csv_out, "wb") as fdst:
                    fdst.write(fsrc.read())
                timed_out = False
                rc = None
                try:
                    proc = subprocess.run(
                        [exe, csv_out, treg_out],
                        env=env, cwd=tmpdir,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        timeout=10,
                    )
                    rc = proc.returncode
                except subprocess.TimeoutExpired:
                    timed_out = True
                if looks_like_crash(rc, timed_out):
                    crashes.append((treg_rel, name, rc, timed_out))
                    print(f"  ★CRASH★ {name}: returncode={rc} timeout={timed_out}")
            print(f"  ({len(variants)}件検証、クラッシュ {sum(1 for c in crashes if c[0] == treg_rel)}件)")

    print(f"\n========================================")
    print(f"検証バリアント数: {total}  クラッシュ検出: {len(crashes)}")
    if crashes:
        print("❌ 以下のバリアントでクラッシュ相当の異常終了を検出しました:")
        for treg_rel, name, rc, timed_out in crashes:
            print(f"  - {treg_rel} / {name}: returncode={rc} timeout={timed_out}")
        return 1
    print("✅ 全バリアントでクラッシュなし(正常終了 or fatal()経由のエラー終了)を確認")
    return 0


if __name__ == "__main__":
    sys.exit(main())
