"""e2e_compare_predictions.py — 中-M4(CODE_REVIEW_2026-07-16.md):
アプリ内予測(predict_template.py)↔.treg予測(native/JS)のE2Eパリティ検証。

これまでのCIは「.tregを読む3実装同士」の閉じた比較のみで、「画面のR²(アプリ内Python予測)と
配布物(.treg)の予測が同じ」という一気通貫の保証がなかった。このスクリプトは
train_bridge.py で実際に学習したモデルに対する3つの出力CSV
  (a) predict_template.py の出力 (アプリ内Python予測)
  (b) native(predict_native_v2.cpp) の出力
  (c) predict-core.js(run_e2e_predict.js) の出力
を読み込み、target列を相対誤差で突き合わせる。

使い方:
    python tests/e2e_compare_predictions.py <target_col> <a.csv> <b.csv> <c.csv>
        [--tol-abs 1e-5] [--tol-rel 1e-4]

終了コード: 全行・全ペアで許容誤差内なら0、1件でも超過があれば1。
"""
import argparse
import math
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def read_target_col(path, target_col):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if target_col not in df.columns:
        raise SystemExit(f"FATAL: {path} に列 '{target_col}' がありません(列一覧: {list(df.columns)})")
    return pd.to_numeric(df[target_col], errors="coerce").values


def compare(name_a, a, name_b, b, tol_abs, tol_rel):
    if len(a) != len(b):
        print(f"FATAL: {name_a}({len(a)}行) と {name_b}({len(b)}行) の行数が一致しません")
        return False
    fail = 0
    max_diff = 0.0
    worst_row = -1
    for i, (va, vb) in enumerate(zip(a, b)):
        a_nan = va is None or (isinstance(va, float) and math.isnan(va))
        b_nan = vb is None or (isinstance(vb, float) and math.isnan(vb))
        if a_nan != b_nan:
            fail += 1
            continue
        if a_nan and b_nan:
            continue
        diff = abs(va - vb)
        tol = max(tol_abs, tol_rel * abs(vb))
        if diff > tol:
            fail += 1
        if diff > max_diff:
            max_diff = diff
            worst_row = i
    status = "PASS" if fail == 0 else "FAIL"
    print(f"  [{status}] {name_a} vs {name_b}: 最大誤差={max_diff:.3e}"
          f"{f' (最悪行 #{worst_row})' if worst_row >= 0 else ''}  不一致={fail}件/{len(a)}行")
    return fail == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target_col")
    ap.add_argument("csv_python", help="predict_template.py の出力CSV(アプリ内Python予測)")
    ap.add_argument("csv_native", help="native(predict_native_v2.cpp) の出力CSV")
    ap.add_argument("csv_js", help="predict-core.js(run_e2e_predict.js) の出力CSV")
    ap.add_argument("--tol-abs", type=float, default=1e-5)
    ap.add_argument("--tol-rel", type=float, default=1e-4)
    args = ap.parse_args()

    y_py = read_target_col(args.csv_python, args.target_col)
    y_native = read_target_col(args.csv_native, args.target_col)
    y_js = read_target_col(args.csv_js, args.target_col)

    print(f"E2Eパリティ検証: target={args.target_col}  行数={len(y_py)}  "
          f"許容誤差=max({args.tol_abs:.1e}, {args.tol_rel:.1e}*|値|)")
    ok = True
    ok &= compare("python(app)", y_py, "native", y_native, args.tol_abs, args.tol_rel)
    ok &= compare("python(app)", y_py, "js", y_js, args.tol_abs, args.tol_rel)
    ok &= compare("native", y_native, "js", y_js, args.tol_abs, args.tol_rel)

    if not ok:
        print("\n❌ アプリ内予測と.treg予測(native/JS)が一致しません。")
        sys.exit(1)
    print("\n✅ アプリ内予測(Python) ↔ .treg予測(native) ↔ .treg予測(JS) が一致しました。")


if __name__ == "__main__":
    main()
