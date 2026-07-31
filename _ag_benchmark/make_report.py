"""_ag_benchmark/make_report.py

treg_records.jsonl + ag_records.jsonl をマージし、中立採点(test集合・同一コード)に基づく
比較を集計して results/aggregate.csv・results/report.md・results/viz_data.json を生成する。
embed python(numpy/pandas)でも AG venv でも動く。
"""
import json
import os

import numpy as np
import pandas as pd

import bench_common as bc

TREG = os.path.join(bc.RESULTS_DIR, "treg_records.jsonl")
AG = os.path.join(bc.RESULTS_DIR, "ag_records.jsonl")
TIE_EPS = 0.005  # test R² 差がこれ未満なら引き分け


def _get(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return d if d is not None else default


def build_rows():
    treg = {r["dataset"]: r for r in bc.read_jsonl(TREG)}
    ag = {r["dataset"]: r for r in bc.read_jsonl(AG)}
    rows = []
    for name in treg:
        t = treg[name]
        a = ag.get(name, {})
        tq = _get(t, "modes", "quick", default={})
        tt = _get(t, "modes", "thorough", default={})
        tt_r2 = tt.get("test_r2")
        ag_r2 = a.get("test_r2")
        gap = (ag_r2 - tt_r2) if (ag_r2 is not None and tt_r2 is not None) else None
        if gap is None:
            winner = "n/a"
        elif gap > TIE_EPS:
            winner = "AG"
        elif gap < -TIE_EPS:
            winner = "treg"
        else:
            winner = "tie"
        rows.append({
            "dataset": name,
            "source": t.get("source"),
            "family": t.get("family"),
            "ceiling_r2": t.get("ceiling_r2"),
            "n_train": t.get("n_train"),
            "n_test": t.get("n_test"),
            "tq_r2": tq.get("test_r2"),
            "tt_r2": tt_r2,
            "tt_self_r2": tt.get("self_r2"),
            "tt_model": tt.get("best_model"),
            "tt_sec": tt.get("train_sec"),
            "tq_sec": tq.get("train_sec"),
            "ag_r2": ag_r2,
            "ag_model": a.get("best_model"),
            "ag_budget": a.get("time_budget_sec"),
            "ag_sec": a.get("fit_sec"),
            "ag_nmodels": a.get("n_models"),
            "gap_ag_minus_tt": round(gap, 4) if gap is not None else None,
            "winner": winner,
            "note": t.get("note"),
        })
    # dataset 名でソート(source, family 順)
    rows.sort(key=lambda r: (r["source"] or "", r["family"] or "", r["dataset"]))
    return rows


def _mean(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return round(float(np.mean(vals)), 4) if vals else None


def _median(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return round(float(np.median(vals)), 4) if vals else None


def summarize(rows):
    paired = [r for r in rows if r["tt_r2"] is not None and r["ag_r2"] is not None]
    wins = {"AG": 0, "treg": 0, "tie": 0}
    for r in paired:
        wins[r["winner"]] = wins.get(r["winner"], 0) + 1
    summ = {
        "n_datasets": len(rows),
        "n_paired": len(paired),
        "mean_tq_r2": _mean([r["tq_r2"] for r in rows]),
        "mean_tt_r2": _mean([r["tt_r2"] for r in rows]),
        "mean_ag_r2": _mean([r["ag_r2"] for r in rows]),
        "median_gap": _median([r["gap_ag_minus_tt"] for r in paired]),
        "mean_gap": _mean([r["gap_ag_minus_tt"] for r in paired]),
        "wins": wins,
        "mean_tt_sec": _mean([r["tt_sec"] for r in rows]),
        "mean_ag_sec": _mean([r["ag_sec"] for r in rows]),
        "mean_tq_sec": _mean([r["tq_sec"] for r in rows]),
    }
    # 自己申告の誠実性(thorough): self_r2 - test_r2 の平均(正=過大申告)
    honesty = [(r["tt_self_r2"] - r["tt_r2"]) for r in rows
               if r["tt_self_r2"] is not None and r["tt_r2"] is not None]
    summ["mean_self_minus_test"] = round(float(np.mean(honesty)), 4) if honesty else None
    # 天井到達率(合成・ceiling>0.1 のみ): test_r2 / ceiling
    def attain(key):
        vs = [r[key] / r["ceiling_r2"] for r in rows
              if r["source"] == "synthetic" and r["ceiling_r2"] and r["ceiling_r2"] > 0.1
              and r[key] is not None]
        return round(float(np.mean(vs)), 4) if vs else None
    summ["attain_tt"] = attain("tt_r2")
    summ["attain_ag"] = attain("ag_r2")
    return summ, paired


def fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def make_markdown(rows, summ):
    L = []
    L.append("# T-regressor バックエンド vs AutoGluon — 30問回帰ベンチマーク\n")
    L.append("## 比較プロトコル(公平性)\n")
    L.append("- **同一の train/test 分割**(固定シード=42, test=25%)を両ツールに与える。treg が書いた"
             "分割ファイルを AG も読むことで分割の同一性を厳密に保証。")
    L.append("- **中立採点**: 両ツールの test 予測を同一コード(純numpy R²/RMSE/MAE)で採点。"
             "treg の自己申告R²(OOF/val)は比較には使わず、別途「誠実性」の参考としてのみ記録。")
    L.append("- **同一の時間バジェット**: AG には各問題の treg *thorough* 実測学習時間を time_limit として与える"
             "(下限60s, preset=good_quality, Windows安定化のためRay不使用・dynamic_stacking無効)。"
             "「同じ時間でどちらが賢いか」の比較。")
    L.append(f"- 合成問題は既知の **ceiling_r2**(どんな学習器も原理的に超えられない test R²上限)を併記。\n")

    w = summ["wins"]
    L.append("## ヘッドライン\n")
    L.append(f"- 対戦成立: **{summ['n_paired']}問**(全{summ['n_datasets']}問)")
    L.append(f"- **勝敗(treg thorough vs AG, 差{TIE_EPS}以内は引分)**: "
             f"AG勝ち **{w.get('AG',0)}** / treg勝ち **{w.get('treg',0)}** / 引分 **{w.get('tie',0)}**")
    L.append(f"- **平均 test R²**: treg quick={fmt(summ['mean_tq_r2'])}, "
             f"treg thorough={fmt(summ['mean_tt_r2'])}, AG={fmt(summ['mean_ag_r2'])}")
    L.append(f"- **AG − treg(thorough)の test R²差**: 中央値={fmt(summ['median_gap'])}, 平均={fmt(summ['mean_gap'])} "
             f"(正ならAG優位)")
    L.append(f"- **平均学習時間**: treg quick={fmt(summ['mean_tq_sec'],1)}s, "
             f"treg thorough={fmt(summ['mean_tt_sec'],1)}s, AG={fmt(summ['mean_ag_sec'],1)}s")
    if summ["attain_tt"] is not None:
        L.append(f"- **天井到達率(合成, test R²/ceiling)**: treg thorough={fmt(summ['attain_tt'])}, "
                 f"AG={fmt(summ['attain_ag'])} (1.0で理論上限)")
    if summ["mean_self_minus_test"] is not None:
        L.append(f"- **treg 自己申告の誠実性(self R² − test R²)**: 平均={fmt(summ['mean_self_minus_test'])} "
                 f"(正なら自己申告が楽観的)")
    L.append("")

    L.append("## 全問結果(test集合・中立採点)\n")
    L.append("| # | dataset | src | family | n_tr/te | ceiling | treg quick | treg thorough | treg model | AG | AG model | 勝者 | AG−tt |")
    L.append("|--:|---|---|---|---|--:|--:|--:|---|--:|---|:--:|--:|")
    for i, r in enumerate(rows, 1):
        L.append(f"| {i} | {r['dataset']} | {r['source'][:4] if r['source'] else '—'} | "
                 f"{r['family'] or '—'} | {r['n_train']}/{r['n_test']} | {fmt(r['ceiling_r2'])} | "
                 f"{fmt(r['tq_r2'])} | {fmt(r['tt_r2'])} | {(r['tt_model'] or '—')[:16]} | "
                 f"{fmt(r['ag_r2'])} | {(r['ag_model'] or '—')[:16]} | {r['winner']} | "
                 f"{fmt(r['gap_ag_minus_tt'])} |")
    L.append("")

    # ソース別集計
    L.append("## ソース別 平均 test R²\n")
    L.append("| source | n | treg quick | treg thorough | AG |")
    L.append("|---|--:|--:|--:|--:|")
    for src in ("synthetic", "real"):
        sub = [r for r in rows if r["source"] == src]
        if sub:
            L.append(f"| {src} | {len(sub)} | {fmt(_mean([r['tq_r2'] for r in sub]))} | "
                     f"{fmt(_mean([r['tt_r2'] for r in sub]))} | {fmt(_mean([r['ag_r2'] for r in sub]))} |")
    L.append("")

    # 大きく負けた/勝った問題
    paired = [r for r in rows if r["gap_ag_minus_tt"] is not None]
    worst = sorted(paired, key=lambda r: r["gap_ag_minus_tt"], reverse=True)[:5]
    best = sorted(paired, key=lambda r: r["gap_ag_minus_tt"])[:5]
    L.append("## treg が最も離された問題(AG−tt 上位)\n")
    for r in worst:
        L.append(f"- **{r['dataset']}** ({r['family']}): tt={fmt(r['tt_r2'])} vs AG={fmt(r['ag_r2'])} "
                 f"(差 {fmt(r['gap_ag_minus_tt'])}) — {r['note']}")
    L.append("\n## treg が競り勝った/並んだ問題(AG−tt 下位)\n")
    for r in best:
        L.append(f"- **{r['dataset']}** ({r['family']}): tt={fmt(r['tt_r2'])} vs AG={fmt(r['ag_r2'])} "
                 f"(差 {fmt(r['gap_ag_minus_tt'])}) — {r['note']}")
    L.append("")
    return "\n".join(L)


def main():
    rows = build_rows()
    summ, paired = summarize(rows)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(bc.RESULTS_DIR, "aggregate.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(bc.RESULTS_DIR, "report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(make_markdown(rows, summ))

    with open(os.path.join(bc.RESULTS_DIR, "viz_data.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"rows": rows, "summary": summ}, f, ensure_ascii=False, indent=1)

    print("=== SUMMARY ===")
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    print(f"\n生成: results/aggregate.csv, results/report.md, results/viz_data.json  ({len(rows)}問)")


if __name__ == "__main__":
    main()
