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

# ── モデル選択・妥当性チェックの閾値(2026-08 追加) ─────────────────────────
# 「結果(R²)の比較だけでなく、意図通りのモデルが選ばれているか」を自動判定するため。
# ATTAIN_MIN: 合成問題(ceiling_r2既知)で「天井にどれだけ近づけたか」の下限。
#   0.1未満のceiling(ノイズに近い問題)は分母が不安定なので対象外。
ATTAIN_MIN = 0.85
ATTAIN_FAMILIES = {"linear", "nonlinear", "categorical", "mixed"}
# pure_noise 等「シグナルなし」問題での過信(リーク疑い)判定閾値
LEAK_ABS_MAX = 0.15
# LGBM の fold バギング variant は同じ「木系」として model_type を正規化して照合する
MODEL_ALIAS = {"lgbm_bag": "lgbm"}


def _norm_model(model_type):
    if not model_type:
        return model_type
    return MODEL_ALIAS.get(model_type, model_type)


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
            "expected_models": t.get("expected_models"),
            "n_train": t.get("n_train"),
            "n_test": t.get("n_test"),
            "tq_r2": tq.get("test_r2"),
            "tt_r2": tt_r2,
            "tt_self_r2": tt.get("self_r2"),
            "tt_train_r2": tt.get("train_r2"),
            "tt_model": tt.get("best_model"),
            "tt_model_type": _norm_model(tt.get("model_type")),
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


def check_model_sanity(rows):
    """「結果(R²)の比較だけでなく、意図通りのモデルが選ばれているか」の自動判定。

    2つの独立したシグナルを見る:
    (a) 天井到達率(attain = test_r2 / ceiling_r2): 合成問題(linear/nonlinear/
        categorical/mixed)でどれだけ理論上限に近づけたか。モデル種別によらず
        性能が出ていれば良しとする(「Linearが交互作用を解けた」等は加点であって
        減点ではない — 詳細は gen_synthetic.py の EXPECTED_MODELS コメント参照)。
    (b) expected_models: 生成関数のnote自体が特定モデル種別を明示的に予見している
        狭いケースのみ判定(該当データセットだけ)。
    (c) pure_noise 等シグナルなし問題は「|test_r2| が小さいか」(過信・リーク検知)。

    「不一致」単独では要確認扱いにしない(Ridgeが交互作用を解けるのは実力であって
    バグではない)。(a)(b)(c)のうち性能面で実際に問題がある場合だけ要確認とする。
    """
    checks = []
    for r in rows:
        fam = r["family"]
        ceil = r["ceiling_r2"]
        tt_r2 = r["tt_r2"]
        model = r["tt_model_type"]
        expected = r["expected_models"]

        attain = None
        perf_flag = None
        if ceil is not None and ceil > 0.1 and fam in ATTAIN_FAMILIES and tt_r2 is not None:
            attain = round(tt_r2 / ceil, 4)
            perf_flag = "OK" if attain >= ATTAIN_MIN else f"天井未達(attain={attain:.2f})"

        leak_flag = None
        if ceil is not None and ceil <= 1e-6 and tt_r2 is not None:
            leak_flag = "OK" if abs(tt_r2) < LEAK_ABS_MAX else f"過信の疑い(|test_r2|={abs(tt_r2):.2f})"

        model_match = None
        if expected:
            model_match = "一致" if model in expected else "不一致(想定外モデル)"

        issues = []
        if perf_flag and perf_flag != "OK":
            issues.append(perf_flag)
        if leak_flag and leak_flag != "OK":
            issues.append(leak_flag)
        # モデル不一致は、性能面でも天井未達の場合だけ要確認に格上げする
        # (性能が出ているのに想定外モデルなのは「別解を見つけた」だけで問題ではない)
        if model_match == "不一致(想定外モデル)" and perf_flag and perf_flag != "OK":
            issues.append("想定モデルと不一致")

        checks.append({
            "dataset": r["dataset"], "family": fam, "note": r["note"],
            "expected_models": expected, "actual_model": model,
            "ceiling_r2": ceil, "attain": attain, "model_match": model_match,
            "status": "OK" if not issues else " / ".join(issues),
        })
    return checks


def _prep_for_generic_model(df, feat_cols, max_onehot_card=20):
    """Ridge/RandomForest/XGBoost/CatBoost用に、object列を軽くone-hot化し数値は中央値補完する。
    (LightGBMには使わず、そちらはcategory dtypeのまま渡して自然な扱いをさせる)"""
    X = df[feat_cols].copy()
    num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in feat_cols if c not in num_cols]
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    keep_cat = [c for c in cat_cols if X[c].nunique(dropna=True) <= max_onehot_card]
    drop_cat = [c for c in cat_cols if c not in keep_cat]
    if drop_cat:
        X = X.drop(columns=drop_cat)
    if keep_cat:
        X = pd.get_dummies(X, columns=keep_cat, dummy_na=True)
    return X


def run_loss_autopsy(rows, top_n=5, gap_min=0.02):
    """diagnose_losses.py/diagnose_losses2.py の手法を標準パイプラインに統合したもの。

    AGにtregが meaningfully 負けた上位datasetについて、同一train/test分割で標準モデル
    (Ridge/RandomForest/LightGBM既定/XGBoost/CatBoost、入手可能なもののみ)を学習し
    test R² を比較する。「モデル族の欠落」か「treg中核(LGBM)の設定力不足」かの
    一次切り分けを毎回自動で行う(個別の深掘りは元のdiagnose_losses*.pyを併用)。
    """
    cands = [r for r in rows if r.get("gap_ag_minus_tt") is not None
             and r["gap_ag_minus_tt"] >= gap_min]
    cands.sort(key=lambda r: r["gap_ag_minus_tt"], reverse=True)
    targets = cands[:top_n]
    if not targets:
        return []

    try:
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        Ridge = RandomForestRegressor = None
    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None
    try:
        import xgboost as xgb
    except ImportError:
        xgb = None
    try:
        import catboost as cb
    except ImportError:
        cb = None

    manifest = bc.load_manifest("manifest_synth.json", "manifest_real.json", "manifest_public.json")
    results = []
    for r in targets:
        name = r["dataset"]
        wd = os.path.join(bc.WORK_DIR, name)
        train_p = os.path.join(wd, "train.csv")
        test_p = os.path.join(wd, "test.csv")
        if not (os.path.exists(train_p) and os.path.exists(test_p)):
            results.append({"dataset": name, "family": r["family"], "tt_r2": r["tt_r2"],
                            "ag_r2": r["ag_r2"], "gap": r["gap_ag_minus_tt"],
                            "error": "分割ファイルなし(要再実行: results/_work/<dataset>/)"})
            continue
        target = manifest.get(name, {}).get("target", "y")
        train_df = bc.read_any_csv(train_p)
        test_df = bc.read_any_csv(test_p)
        if target not in train_df.columns:
            results.append({"dataset": name, "family": r["family"], "tt_r2": r["tt_r2"],
                            "ag_r2": r["ag_r2"], "gap": r["gap_ag_minus_tt"],
                            "error": f"target列'{target}'が見当たらない"})
            continue
        feat_cols = [c for c in train_df.columns if c != target]
        y_tr = train_df[target].values
        y_te = test_df[target].values

        scores = {}
        try:
            Xg_tr = _prep_for_generic_model(train_df, feat_cols)
            Xg_te = _prep_for_generic_model(test_df, feat_cols).reindex(columns=Xg_tr.columns, fill_value=0)
        except Exception as e:
            Xg_tr = Xg_te = None
            scores["前処理"] = f"ERR:{str(e)[:80]}"

        if Xg_tr is not None:
            if Ridge is not None:
                try:
                    m = Ridge(alpha=1.0).fit(Xg_tr.values, y_tr)
                    r2, *_ = bc.score(y_te, m.predict(Xg_te.values))
                    scores["Ridge"] = r2
                except Exception as e:
                    scores["Ridge"] = f"ERR:{str(e)[:60]}"
            if RandomForestRegressor is not None:
                try:
                    m = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(Xg_tr.values, y_tr)
                    r2, *_ = bc.score(y_te, m.predict(Xg_te.values))
                    scores["RandomForest"] = r2
                except Exception as e:
                    scores["RandomForest"] = f"ERR:{str(e)[:60]}"
            if xgb is not None:
                try:
                    m = xgb.XGBRegressor(n_estimators=500, random_state=42, verbosity=0).fit(Xg_tr.values, y_tr)
                    r2, *_ = bc.score(y_te, m.predict(Xg_te.values))
                    scores["XGBoost"] = r2
                except Exception as e:
                    scores["XGBoost"] = f"ERR:{str(e)[:60]}"
            if cb is not None:
                try:
                    m = cb.CatBoostRegressor(iterations=500, verbose=False, random_state=42).fit(Xg_tr.values, y_tr)
                    r2, *_ = bc.score(y_te, m.predict(Xg_te.values))
                    scores["CatBoost"] = r2
                except Exception as e:
                    scores["CatBoost"] = f"ERR:{str(e)[:60]}"

        if lgb is not None:
            try:
                Xl_tr = train_df[feat_cols].copy()
                Xl_te = test_df[feat_cols].copy()
                for c in feat_cols:
                    if not pd.api.types.is_numeric_dtype(Xl_tr[c]):
                        Xl_tr[c] = Xl_tr[c].astype("category")
                        Xl_te[c] = pd.Categorical(Xl_te[c], categories=Xl_tr[c].cat.categories)
                m = lgb.LGBMRegressor(n_estimators=500, verbosity=-1, random_state=42).fit(Xl_tr, y_tr)
                r2, *_ = bc.score(y_te, m.predict(Xl_te))
                scores["LightGBM(既定)"] = r2
            except Exception as e:
                scores["LightGBM(既定)"] = f"ERR:{str(e)[:60]}"

        results.append({
            "dataset": name, "family": r["family"], "tt_r2": r["tt_r2"], "ag_r2": r["ag_r2"],
            "gap": r["gap_ag_minus_tt"], "scores": scores,
        })
    return results


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


def make_markdown(rows, summ, sanity=None, autopsy=None):
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
    L.append(f"- 合成問題は既知の **ceiling_r2**(どんな学習器も原理的に超えられない test R²上限)を併記。"
             f"ただしこれは全データでの理論値であり、test集合(25%)は有限サンプルのため、"
             f"稀に test_r2 が ceiling_r2 をわずかに上回ることがある(サンプリング分散によるもので"
             f"異常ではない。例: count_poisson)。\n")

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

    # ── モデル選択・妥当性チェック(2026-08追加) ───────────────────────────
    if sanity:
        n_ok = sum(1 for c in sanity if c["status"] == "OK")
        n_ng = len(sanity) - n_ok
        L.append("## モデル選択・妥当性チェック\n")
        L.append("結果(R²)の比較だけでなく、「意図通りにモデルが選ばれ、意図通りの性能が"
                 "出ているか」を自動判定した一覧。判定基準は2つ: (a) 天井到達率"
                 f"(test_r2/ceiling_r2 が {ATTAIN_MIN} 未満なら要確認)、"
                 "(b) 生成関数のnoteが特定モデル種別を明示的に予見しているケースのみ、"
                 "実際に選ばれたモデルと突き合わせる(該当しないデータセットは判定対象外)。"
                 "**モデルの不一致だけでは要確認にしない**(性能が出ているなら、"
                 "想定と違うモデルで解けたのは問題ではなく別解と見なす)。\n")
        L.append(f"- **判定結果**: OK **{n_ok}** / 要確認 **{n_ng}** (全{len(sanity)}問中、判定対象外は除く)\n")
        L.append("| dataset | family | 想定モデル | 実際のモデル | ceiling | attain | 判定 |")
        L.append("|---|---|---|---|--:|--:|---|")
        for c in sanity:
            if c["expected_models"] is None and c["attain"] is None:
                continue  # 判定対象外(参考情報もない)はテーブルから省く
            exp = ",".join(c["expected_models"]) if c["expected_models"] else "—"
            L.append(f"| {c['dataset']} | {c['family'] or '—'} | {exp} | "
                     f"{c['actual_model'] or '—'} | {fmt(c['ceiling_r2'])} | "
                     f"{fmt(c['attain'])} | {c['status']} |")
        L.append("")
        flagged = [c for c in sanity if c["status"] != "OK"]
        if flagged:
            L.append("### 要確認の詳細\n")
            for c in flagged:
                L.append(f"- **{c['dataset']}** ({c['family']}): {c['status']} — {c['note']}")
            L.append("")

    # ── 個別調査(loss autopsy, diagnose_losses系の標準統合, 2026-08追加) ──────
    if autopsy:
        L.append("## AG優位問題の個別調査(標準モデル比較による一次切り分け)\n")
        L.append("AGにtregが明確に負けた上位問題について、同一train/test分割で標準的な"
                 "単体モデル(Ridge/RandomForest/LightGBM既定設定/XGBoost/CatBoost、"
                 "入手可能なもののみ)を学習し test R² を比較する。treg中核(LGBM+Ridge/GP/MLP"
                 "のBlend)が同格の単体モデルにも負けているなら「treg設定の弱さ」、"
                 "CatBoost/XGBoost等treg未搭載の族だけが勝つなら「モデル族の欠落」、"
                 "単体はどれも横並びでAGの stacked ensemble だけ勝つなら「スタッキングの深さ」が"
                 "示唆される(詳細な深掘りは diagnose_losses.py / diagnose_losses2.py を参照)。\n")
        for a in autopsy:
            if a.get("error"):
                L.append(f"- **{a['dataset']}**: {a['error']}")
                continue
            L.append(f"### {a['dataset']} ({a.get('family') or '—'}) — "
                     f"treg={fmt(a['tt_r2'])} / AG={fmt(a['ag_r2'])} (差 {fmt(a['gap'])})\n")
            L.append("| 標準モデル | test R² |")
            L.append("|---|--:|")
            for mname, sc in (a.get("scores") or {}).items():
                sc_disp = fmt(sc) if isinstance(sc, (int, float)) else str(sc)
                L.append(f"| {mname} | {sc_disp} |")
            L.append("")

    return "\n".join(L)


def main():
    import sys
    skip_autopsy = "--skip-autopsy" in sys.argv
    rows = build_rows()
    summ, paired = summarize(rows)
    sanity = check_model_sanity(rows)
    autopsy = [] if skip_autopsy else run_loss_autopsy(rows)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(bc.RESULTS_DIR, "aggregate.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(bc.RESULTS_DIR, "report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(make_markdown(rows, summ, sanity=sanity, autopsy=autopsy))

    with open(os.path.join(bc.RESULTS_DIR, "viz_data.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"rows": rows, "summary": summ, "model_sanity": sanity, "loss_autopsy": autopsy},
                  f, ensure_ascii=False, indent=1)

    n_ok = sum(1 for c in sanity if c["status"] == "OK")
    print("=== SUMMARY ===")
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    print(f"\n[モデル選択・妥当性チェック] OK {n_ok} / 要確認 {len(sanity) - n_ok} (全{len(sanity)}問)")
    print(f"生成: results/aggregate.csv, results/report.md, results/viz_data.json  ({len(rows)}問)")


if __name__ == "__main__":
    main()
