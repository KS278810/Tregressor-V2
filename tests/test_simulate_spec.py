# -*- coding: utf-8 -*-
"""③ SIMULATE 用データ契約(slider_spec / y_hist / seed_rows / neighbor_ref / corr_pairs)
の単体テスト。lightgbm/sklearn を必要としないよう、train_bridge の該当ヘルパーだけを
インポートして検証する。設計: docs/interactive-predict-design.md §10.2"""
import os, sys, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_bridge as tb

FAIL = []
def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)

def make_df(n=600, seed=7):
    rs = np.random.RandomState(seed)
    temperature = 22 + 6.5 * rs.randn(n)
    pressure = 93.5 + 0.34 * temperature + 1.2 * rs.randn(n)       # r≈0.88
    flow = np.maximum(0.5, 14 + 4 * rs.randn(n))
    city = rs.choice(['Tokyo', 'Osaka', 'Nagoya'], n)
    machine = rs.choice(['A', 'B'], n)
    count = rs.randint(0, 20, n)                                   # 整数列
    y = (11 + 1.35 * temperature + 0.42 * pressure - 0.16 * (flow - 15.5) ** 2
         + np.where(city == 'Osaka', -3.4, 0.0) + 2.0 * rs.randn(n))
    df = pd.DataFrame({'temperature': temperature, 'pressure': pressure,
                       'flow_rate': flow, 'count': count,
                       'city': city, 'machine': machine, 'yield': y})
    df.loc[df.index[:5], 'temperature'] = np.nan                   # 欠損混入
    return df

def main():
    df = make_df()
    target = 'yield'
    raw_cols = ['temperature', 'pressure', 'flow_rate', 'count', 'city', 'machine']
    num_cols = ['temperature', 'pressure', 'flow_rate', 'count']

    print("[1] 重要度の生列集約")
    onehot = [{'feature_name': 'city=Osaka', 'source_col': 'city', 'class_value': 'Osaka'},
              {'feature_name': 'city=Tokyo', 'source_col': 'city', 'class_value': 'Tokyo'}]
    dtspec = [{'feature_name': 'ts__hour', 'source_col': 'ts', 'method': 'datetime', 'part': 'hour'}]
    derived = [{'name': 'temperature*pressure', 'op': 'mul', 'cols': ['temperature', 'pressure']},
               {'name': 'flow_rate^2', 'op': 'sq', 'cols': ['flow_rate']}]
    feat_full = [
        {'name': 'temperature', 'pct': 30.0},
        {'name': 'city=Osaka', 'pct': 10.0},
        {'name': 'city=Tokyo', 'pct': 6.0},
        {'name': 'temperature*pressure', 'pct': 20.0},   # → temperature/pressure に等分
        {'name': 'flow_rate^2', 'pct': 8.0},
        {'name': 'temperature pressure', 'pct': 12.0},   # poly 交互作用
        {'name': 'flow_rate^2', 'pct': 0.0},
        {'name': 'machine', 'pct': 14.0},
        {'name': 'ts__hour', 'pct': 0.0},
        {'name': '__unknown_feature__', 'pct': 5.0},     # 由来不明 → 捨てる
    ]
    imp = tb._sim_aggregate_importance(feat_full, raw_cols, onehot, dtspec, None, None, derived)
    check(abs(sum(imp.values()) - 100.0) < 1e-6, f"合計100%に正規化 (実測 {sum(imp.values()):.4f})")
    check(imp.get('city', 0) > 0, "one-hot 2列が city に合算された")
    check(imp.get('pressure', 0) > 0, "派生/polyから pressure に配分された")
    check('__unknown_feature__' not in imp, "由来不明の特徴量は集約先を持たない")
    # 集約先を持つ寄与の合計 = 30+10+6+20+8+12+14 = 100（pct=0 と由来不明は除外される）
    resolved = 30.0 + 10.0 + 6.0 + 20.0 + 8.0 + 12.0 + 14.0
    # temperature = 30(直接) + 20/2(派生 mul) + 12/2(poly 交互作用) = 46
    exp_temp = (30.0 + 20.0 / 2 + 12.0 / 2) / resolved * 100.0
    check(abs(imp['temperature'] - exp_temp) < 1e-6,
          f"temperature の配分が等分規則どおり (実測 {imp['temperature']:.3f}% / 期待 {exp_temp:.3f}%)")
    exp_city = (10.0 + 6.0) / resolved * 100.0
    check(abs(imp['city'] - exp_city) < 1e-6,
          f"city は one-hot 2列の合算 (実測 {imp['city']:.3f}% / 期待 {exp_city:.3f}%)")
    exp_press = (20.0 / 2 + 12.0 / 2) / resolved * 100.0
    check(abs(imp['pressure'] - exp_press) < 1e-6,
          f"pressure は派生とpolyからの配分のみ (実測 {imp['pressure']:.3f}%)")

    print("[2] slider_spec の妥当性")
    x_clip = {'temperature': (8.0, 36.0), 'pressure': (98.0, 106.0)}
    spec = tb._sim_build_slider_spec(df, raw_cols, imp, x_clip)
    check(len(spec) == len(raw_cols), f"全ての生列が1本ずつ ({len(spec)})")
    check(all(spec[i]['importance_pct'] >= spec[i + 1]['importance_pct']
              for i in range(len(spec) - 1)), "重要度の降順に並んでいる")
    for s in spec:
        if s['kind'] != 'numeric':
            continue
        c = s['col']
        check(s['min'] <= s['p1'] + 1e-9 <= s['median'] + 1e-9 <= s['p99'] + 1e-9
              and s['p99'] <= s['max'] + 1e-9, f"{c}: min<=p1<=median<=p99<=max")
        n_finite = int(np.isfinite(pd.to_numeric(df[c], errors='coerce').values).sum())
        check(sum(s['hist']) == n_finite, f"{c}: hist合計 {sum(s['hist'])} == 有限値 {n_finite}")
        check(s['step'] > 0, f"{c}: step > 0")
    sp = {s['col']: s for s in spec}
    check(sp['count']['is_integer'] and sp['count']['step'] == 1.0, "整数列は step=1")
    check(not sp['temperature']['is_integer'], "実数列は is_integer=False")
    check(sp['temperature']['x_clip_lo'] == 8.0, "x_clip が生列に1:1で載る")
    check(sp['flow_rate']['x_clip_lo'] is None, "x_clip の無い列は None（誤った境界を見せない）")
    check(sp['city']['kind'] == 'categorical'
          and [l['value'] for l in sp['city']['levels']]
              == list(df['city'].value_counts().index[:3]), "カテゴリ水準は頻度降順")
    check(sp['city']['mode'] == df['city'].mode()[0], "mode が最頻値")

    print("[3] y_hist")
    y = df[target].values
    yh = tb._sim_build_y_hist(y)
    check(sum(yh['counts']) == yh['n'] == len(y), f"counts合計=={yh['n']}==行数")
    check(tb.SIM_MIN_YHIST_BINS <= len(yh['counts']) <= tb.SIM_MAX_YHIST_BINS,
          f"bin数が範囲内 ({len(yh['counts'])})")
    check(len(yh['bin_edges']) == len(yh['counts']) + 1, "bin_edges の本数が整合")
    check(yh['p10'] <= yh['p50'] <= yh['p90'], "分位が単調")

    print("[4] seed_rows（実在する行であること）")
    seeds = tb._sim_build_seed_rows(df, target, [s['col'] for s in spec])
    check(len(seeds) == 4, "low/median/high/random の4件")
    check(all(isinstance(v, str) for s in seeds for v in s['values'].values()),
          "values は全て文字列（rawRow にそのまま渡せる）")
    ys = {s['label']: s['y'] for s in seeds}
    check(ys['low'] <= ys['median'] <= ys['high'], "low<=median<=high")
    for s in seeds:
        m = (np.abs(pd.to_numeric(df[target], errors='coerce').values - s['y']) < 1e-6)
        check(bool(m.any()), f"{s['label']}: y が実データに存在する")
        i = int(np.where(m)[0][0])
        check(str(df['city'].iloc[i]) == s['values']['city'],
              f"{s['label']}: 同一行の値が揃っている（合成行ではない）")

    print("[5] neighbor_ref（重み付き距離）")
    ref = tb._sim_build_neighbor_ref(df, num_cols, imp)
    check(ref is not None and len(ref['rows']) <= tb.SIM_NEIGHBOR_ROWS,
          f"サンプル行 {len(ref['rows'])} <= {tb.SIM_NEIGHBOR_ROWS}")
    check(len(ref['cols']) == len(ref['mean']) == len(ref['std']) == len(ref['weight']),
          "cols/mean/std/weight の長さが一致")
    check(abs(sum(ref['weight']) - len(ref['cols'])) < 0.05, "weight は平均1に正規化")
    check(ref['radius'] > 0, "radius > 0")

    Z = np.array(ref['rows']); W = np.array(ref['weight'])
    mean = np.array(ref['mean']); std = np.array(ref['std'])
    def neigh(vals):
        z = (np.array([vals[c] for c in ref['cols']]) - mean) / std
        return int((np.sqrt((W * (Z - z) ** 2).sum(axis=1)) <= ref['radius']).sum())
    med = {c: float(np.nanmedian(pd.to_numeric(df[c], errors='coerce'))) for c in ref['cols']}
    real = df.dropna(subset=ref['cols']).iloc[0]
    n_real = neigh({c: float(real[c]) for c in ref['cols']})
    wild = {c: float(pd.to_numeric(df[c], errors='coerce').max()) for c in ref['cols']}
    n_wild = neigh(wild)
    tonly = dict(med); tonly['temperature'] = float(pd.to_numeric(df['temperature']).max())
    n_tonly = neigh(tonly)
    print(f"       実データ行 {n_real} 件 / 全列最大 {n_wild} 件 / temp単独最大 {n_tonly} 件")
    check(n_real > 0, "実在する行は近傍を持つ（次元の呪いを回避できている）")
    check(n_wild == 0, "全列最大の非現実的な組み合わせは 0 件")
    check(n_tonly < n_real, "相関を無視した1変数操作は近傍が減る（ceteris paribus 検出）")

    print("[6] corr_pairs")
    pairs = tb._sim_build_corr_pairs(df, num_cols)
    check(len(pairs) >= 1, f"強相関ペアを検出 ({len(pairs)})")
    tp = [p for p in pairs if {p['a'], p['b']} == {'temperature', 'pressure'}]
    check(bool(tp), "temperature~pressure を検出")
    if tp:
        p = tp[0]
        check(abs(p['r']) >= tb.SIM_CORR_MIN, f"r={p['r']} が閾値以上")
        a = pd.to_numeric(df['temperature']); b = pd.to_numeric(df['pressure'])
        ok = a.notna() & b.notna()
        pred_b = p['sAB'] * a[ok] + p['iAB'] if p['a'] == 'temperature' else p['sBA'] * a[ok] + p['iBA']
        check(float(np.sqrt(((b[ok] - pred_b) ** 2).mean())) < b[ok].std(),
              "回帰係数で相手を予測すると残差が標準偏差より小さい")
    check(all(abs(pairs[i]['r']) >= abs(pairs[i + 1]['r']) for i in range(len(pairs) - 1)),
          "|r| 降順に並んでいる")

    print("[7] 退化ケース")
    check(tb._sim_build_neighbor_ref(df, [], imp) is None, "数値列ゼロ → neighbor_ref は None")
    check(tb._sim_build_corr_pairs(df, ['temperature']) == [], "数値列1本 → corr_pairs は空")
    const_df = pd.DataFrame({'a': [1.0] * 20, 'yield': np.arange(20.0)})
    cspec = tb._sim_build_slider_spec(const_df, ['a'], {'a': 100.0}, {})
    check(len(cspec) == 1 and cspec[0]['max'] > cspec[0]['min'], "定数列でも min<max を保つ")
    check(tb._sim_build_y_hist(np.array([np.nan, np.inf])) is None, "有限値ゼロ → y_hist は None")
    nan_df = df.copy(); nan_df.loc[:, 'city'] = np.nan
    nspec = tb._sim_build_slider_spec(nan_df, ['city'], {'city': 100.0}, {})
    check(nspec[0]['levels'][0]['value'] == tb.CAT_NAN_SENTINEL,
          "全欠損のカテゴリ列は CAT_NAN_SENTINEL になる")

    print()
    if FAIL:
        print(f"NG: {len(FAIL)} 件失敗")
        for m in FAIL:
            print("  - " + m)
        return 1
    print("すべて成功")
    return 0

if __name__ == '__main__':
    sys.exit(main())
