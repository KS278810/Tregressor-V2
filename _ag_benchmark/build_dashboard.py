"""_ag_benchmark/build_dashboard.py

results/viz_data.json(make_report.py 生成)を読み、自己完結の results/dashboard.html を出力する。
- テーマ対応(light/dark トークン + prefers-color-scheme + data-theme)
- 検証済み dataviz 既定パレット: treg=blue(slot1) / AG=orange(slot6) / ceiling=中立グレー
- チャートは静的 SVG(CSSクラスでテーマ対応、<title>でホバー)+ 全問テーブル
numpy/pandas 不要(標準ライブラリのみ)。embed python でも AG venv でも動く。
"""
import html
import json
import os

import bench_common as bc

VIZ = os.path.join(bc.RESULTS_DIR, "viz_data.json")
OUT = os.path.join(bc.RESULTS_DIR, "dashboard.html")


def esc(s):
    return html.escape(str(s), quote=True)


def fnum(v, nd=3):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}"
    return esc(v)


# ─────────────────────────── SVG: gap diverging bars ───────────────────────────
def svg_gap(rows):
    paired = [r for r in rows if r.get("gap_ag_minus_tt") is not None]
    paired.sort(key=lambda r: r["gap_ag_minus_tt"])  # treg有利(負)→上、AG有利(正)→下
    if not paired:
        return '<p class="muted">AG の結果待ち(対戦成立問題なし)。</p>'
    n = len(paired)
    row_h, top, bot, left, right = 22, 16, 24, 210, 40
    plot_w = 560
    W = left + plot_w + right
    H = top + bot + n * row_h
    gmax = max(0.02, max(abs(r["gap_ag_minus_tt"]) for r in paired))
    cx = left + plot_w / 2
    half = plot_w / 2 - 6

    def x_of(g):
        return cx + (g / gmax) * half

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
             f'aria-label="データセット別 AG−treg(thorough) の test R² 差">']
    # 中央ゼロ線
    parts.append(f'<line x1="{cx:.1f}" y1="{top-6}" x2="{cx:.1f}" y2="{H-bot+4}" class="axis"/>')
    parts.append(f'<text x="{cx:.1f}" y="{top-8}" class="tick mid" text-anchor="middle">0(互角)</text>')
    parts.append(f'<text x="{left:.1f}" y="{top-8}" class="tick" text-anchor="start">← treg 有利</text>')
    parts.append(f'<text x="{left+plot_w:.1f}" y="{top-8}" class="tick" text-anchor="end">AG 有利 →</text>')
    for i, r in enumerate(paired):
        y = top + i * row_h
        yc = y + row_h / 2
        g = r["gap_ag_minus_tt"]
        x1 = x_of(0)
        x2 = x_of(g)
        bx, bw = (min(x1, x2), abs(x2 - x1))
        cls = "bar-ag" if g > 0 else "bar-treg"
        name = esc(r["dataset"])
        tip = f'{name}: treg {fnum(r.get("tt_r2"))} vs AG {fnum(r.get("ag_r2"))}(差 {g:+.3f})'
        parts.append(f'<rect x="{bx:.1f}" y="{y+4:.1f}" width="{max(bw,1):.1f}" height="{row_h-8}" '
                     f'rx="3" class="{cls}"><title>{tip}</title></rect>')
        parts.append(f'<text x="{left-8}" y="{yc+4:.1f}" class="rowlab" text-anchor="end">{name}</text>')
        # ラベルがプロット外(左マージンのdataset名)に被る場合はバー内側(白文字)に置く
        label_w = 8 * len(f"{g:+.3f}") + 8
        if g >= 0:
            lx, anc, val_cls = x2 + 6, "start", "pos"
        elif x2 - label_w < left:
            lx, anc, val_cls = x2 + 6, "start", "on-bar"
        else:
            lx, anc, val_cls = x2 - 6, "end", "neg"
        parts.append(f'<text x="{lx:.1f}" y="{yc+4:.1f}" class="val {val_cls}" '
                     f'text-anchor="{anc}">{g:+.3f}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ─────────────────────────── SVG: scatter treg vs AG ───────────────────────────
def svg_scatter(rows):
    pts = [r for r in rows if r.get("tt_r2") is not None and r.get("ag_r2") is not None]
    if not pts:
        return '<p class="muted">AG の結果待ち。</p>'
    S, pad = 380, 44
    W = H = S + pad * 2
    lo = min(0.0, min(min(r["tt_r2"], r["ag_r2"]) for r in pts))
    lo = max(-0.2, lo)  # 下限をほどほどに

    def sx(v):
        return pad + (v - lo) / (1 - lo) * S

    def sy(v):
        return pad + S - (v - lo) / (1 - lo) * S

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
             f'aria-label="treg thorough(横) vs AutoGluon(縦)の test R² 散布図">']
    # グリッド + 目盛
    ticks = [t / 10 for t in range(int(lo * 10) if lo < 0 else 0, 11, 2)]
    for t in ticks:
        parts.append(f'<line x1="{sx(t):.1f}" y1="{pad}" x2="{sx(t):.1f}" y2="{pad+S}" class="grid"/>')
        parts.append(f'<line x1="{pad}" y1="{sy(t):.1f}" x2="{pad+S}" y2="{sy(t):.1f}" class="grid"/>')
        parts.append(f'<text x="{sx(t):.1f}" y="{pad+S+16}" class="tick" text-anchor="middle">{t:.1f}</text>')
        parts.append(f'<text x="{pad-8}" y="{sy(t)+4:.1f}" class="tick" text-anchor="end">{t:.1f}</text>')
    # 互角の対角線
    parts.append(f'<line x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(1):.1f}" y2="{sy(1):.1f}" class="diag"/>')
    parts.append(f'<text x="{sx(1)-4:.1f}" y="{sy(1)+14:.1f}" class="tick" text-anchor="end">互角ライン</text>')
    # 点(source で色分け)
    for r in pts:
        cx, cy = sx(r["tt_r2"]), sy(r["ag_r2"])
        cls = "pt-synth" if r.get("source") == "synthetic" else "pt-real"
        tip = (f'{esc(r["dataset"])}: treg {fnum(r["tt_r2"])} / AG {fnum(r["ag_r2"])}'
               f'(上=AG優位 / 下=treg優位)')
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" class="{cls}"><title>{tip}</title></circle>')
    parts.append(f'<text x="{pad+S/2:.1f}" y="{H-6}" class="axlab" text-anchor="middle">treg thorough  test R² →</text>')
    parts.append(f'<text x="14" y="{pad+S/2:.1f}" class="axlab" text-anchor="middle" '
                 f'transform="rotate(-90 14 {pad+S/2:.1f})">AutoGluon  test R² →</text>')
    parts.append("</svg>")
    return "".join(parts)


# ─────────────────────────── テーブル ───────────────────────────
def table(rows):
    head = ["#", "dataset", "src", "family", "ceiling", "treg quick", "treg thorough",
            "AG", "勝者", "treg model", "AG model", "tt秒", "AG秒"]
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    trs = []
    for i, r in enumerate(rows, 1):
        w = r.get("winner")
        pill = {"AG": "pill-ag", "treg": "pill-treg", "tie": "pill-tie"}.get(w, "pill-na")
        wtxt = {"AG": "AG", "treg": "treg", "tie": "互角"}.get(w, "—")
        src_label = {"synthetic": "合成", "real": "実データ", "public_uci": "UCI"}.get(r.get("source"), "—")
        cells = [
            str(i), esc(r["dataset"]), esc(src_label), esc(r.get("family") or "—"),
            fnum(r.get("ceiling_r2")), fnum(r.get("tq_r2")), fnum(r.get("tt_r2")),
            fnum(r.get("ag_r2")), f'<span class="pill {pill}">{wtxt}</span>',
            esc((r.get("tt_model") or "—")), esc((r.get("ag_model") or "—")),
            fnum(r.get("tt_sec"), 0), fnum(r.get("ag_sec"), 0),
        ]
        td = "".join(f"<td>{c}</td>" for c in cells)
        trs.append(f"<tr>{td}</tr>")
    return f'<table class="grid"><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table>'


def tile(label, value, sub=""):
    return (f'<div class="tile"><div class="tile-l">{esc(label)}</div>'
            f'<div class="tile-v">{value}</div><div class="tile-s">{sub}</div></div>')


def summary_tiles(s):
    w = s.get("wins", {})
    tiles = []
    tiles.append(tile("平均 test R²(thorough)",
                      f'<span class="hl-treg">{fnum(s.get("mean_tt_r2"))}</span> '
                      f'<span class="vs">vs</span> <span class="hl-ag">{fnum(s.get("mean_ag_r2"))}</span>',
                      "treg thorough vs AutoGluon(同一test集合)"))
    tiles.append(tile("勝敗(thorough)",
                      f'{w.get("treg",0)}<span class="vs"> 勝 / </span>{w.get("tie",0)}'
                      f'<span class="vs"> 分 / </span>{w.get("AG",0)}<span class="vs"> 敗</span>',
                      f'差 {bc.TIE_EPS if hasattr(bc,"TIE_EPS") else 0.005} 以内は互角'))
    at, aa = s.get("attain_tt"), s.get("attain_ag")
    tiles.append(tile("天井到達率(合成)",
                      f'<span class="hl-treg">{(f"{at*100:.1f}%" if at else "—")}</span> '
                      f'<span class="vs">vs</span> <span class="hl-ag">{(f"{aa*100:.1f}%" if aa else "—")}</span>',
                      "test R² ÷ 理論上限(1.0=上限到達)"))
    tiles.append(tile("平均学習時間",
                      f'<span class="hl-treg">{fnum(s.get("mean_tt_sec"),0)}s</span> '
                      f'<span class="vs">vs</span> <span class="hl-ag">{fnum(s.get("mean_ag_sec"),0)}s</span>',
                      "thorough vs AG実測(AGはnominal予算を超過しがち)"))
    sm = s.get("mean_self_minus_test")
    tiles.append(tile("treg quick 平均R²", fnum(s.get("mean_tq_r2")),
                      f'自己申告の楽観度(self−test): {fnum(sm)}' if sm is not None else "高速モード"))
    return '<div class="tiles">' + "".join(tiles) + "</div>"


PALETTE_CSS = """
:root{
  --plane:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --treg:#2a78d6; --ag:#eb6834; --treg-soft:rgba(42,120,214,0.14); --ag-soft:rgba(235,104,52,0.14);
  --good:#0ca30c;
}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){
  --plane:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --treg:#3987e5; --ag:#d95926; --treg-soft:rgba(57,135,229,0.20); --ag-soft:rgba(217,89,38,0.20);
  --good:#0ca30c;
}}
:root[data-theme="dark"]{
  --plane:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --treg:#3987e5; --ag:#d95926; --treg-soft:rgba(57,135,229,0.20); --ag-soft:rgba(217,89,38,0.20);
}
"""


def build(rows, summ):
    css = PALETTE_CSS + """
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.5;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:40px 24px 80px}
header{border-bottom:1px solid var(--border);padding-bottom:20px;margin-bottom:28px}
.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);
  font-weight:600;margin:0 0 8px}
h1{font-size:30px;line-height:1.15;margin:0 0 10px;text-wrap:balance;font-weight:700}
.lede{color:var(--ink2);max-width:64ch;margin:0;font-size:15px}
.legend{display:flex;gap:18px;margin-top:16px;flex-wrap:wrap;font-size:13px;color:var(--ink2)}
.legend b{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:middle}
.b-treg{background:var(--treg)} .b-ag{background:var(--ag)}
h2{font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);
  font-weight:700;margin:44px 0 14px;padding-bottom:6px;border-bottom:1px solid var(--border)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 18px}
.tile-l{font-size:12px;color:var(--muted);font-weight:600;margin-bottom:8px}
.tile-v{font-size:24px;font-weight:700;letter-spacing:-.01em}
.tile-s{font-size:11.5px;color:var(--ink2);margin-top:8px;line-height:1.4}
.hl-treg{color:var(--treg)} .hl-ag{color:var(--ag)} .vs{color:var(--muted);font-weight:400;font-size:.7em}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:20px;
  overflow-x:auto}
.chart{display:block;width:100%;height:auto;min-width:520px}
.axis{stroke:var(--axis);stroke-width:1}
.grid{stroke:var(--grid);stroke-width:1}
.diag{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:5 4}
.bar-treg{fill:var(--treg)} .bar-ag{fill:var(--ag)}
.pt-synth{fill:var(--treg);stroke:var(--surface);stroke-width:1.5}
.pt-real{fill:var(--ag);stroke:var(--surface);stroke-width:1.5}
.rowlab{fill:var(--ink2);font-size:11.5px}
.tick{fill:var(--muted);font-size:10.5px} .tick.mid{fill:var(--ink2);font-weight:600}
.axlab{fill:var(--ink2);font-size:12px;font-weight:600}
.val{font-size:10.5px;font-weight:600;font-variant-numeric:tabular-nums}
.val.pos{fill:var(--ag)} .val.neg{fill:var(--treg)} .val.on-bar{fill:#fff}
.muted{color:var(--muted)}
table.grid{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
table.grid th,table.grid td{padding:7px 9px;text-align:right;border-bottom:1px solid var(--border);
  white-space:nowrap}
table.grid th:nth-child(2),table.grid td:nth-child(2),
table.grid th:nth-child(4),table.grid td:nth-child(4),
table.grid th:nth-child(10),table.grid td:nth-child(10),
table.grid th:nth-child(11),table.grid td:nth-child(11){text-align:left}
table.grid th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;
  letter-spacing:.04em;position:sticky;top:0;background:var(--surface)}
table.grid tbody tr:hover{background:var(--treg-soft)}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:600}
.pill-treg{background:var(--treg-soft);color:var(--treg)}
.pill-ag{background:var(--ag-soft);color:var(--ag)}
.pill-tie{background:rgba(137,135,129,.18);color:var(--ink2)}
.pill-na{color:var(--muted)}
.note{font-size:12.5px;color:var(--ink2);background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:14px 16px;margin-top:14px;line-height:1.55}
footer{margin-top:40px;font-size:12px;color:var(--muted);border-top:1px solid var(--border);padding-top:16px}
"""
    n = summ.get("n_datasets", len(rows))
    npair = summ.get("n_paired", 0)
    body = f"""<div class="wrap">
<header>
  <p class="eyebrow">Backend capability benchmark</p>
  <h1>T-regressor バックエンド vs AutoGluon</h1>
  <p class="lede">{n} 問の回帰タスクを<strong>共通のホールドアウト test 集合</strong>で対決。
  内訳は合成24問(構造既知・理論上限R²付き)+ UCI/sklearn実データ6問 +
  <strong>UCI ML Repository公開ベンチマーク10問</strong>(concrete/energy/power plant/parkinsons/
  superconductivity等、OpenML-CTR23・AutoML-Benchmarkが表形式回帰の学術標準として重視するのと
  同種の、出典明記された著名データセット)。両ツールとも同一の train/test 分割で学習し、
  <strong>同一の中立採点コード</strong>(純numpy R²)で採点。AutoGluon には各問題の treg thorough
  実測学習時間を time_limit として与えた(同時間対決)。対戦成立 {npair} 問。</p>
  <div class="legend">
    <span><b class="b-treg"></b>T-regressor(thorough)</span>
    <span><b class="b-ag"></b>AutoGluon(good_quality)</span>
    <span>散布図: ● 合成 / ● 実データ(UCI含む)</span>
  </div>
  <p class="note" style="margin-top:16px">本ベンチで判明した最大の敗北 <code>real_cpu</code>(旧 treg 0.78 vs AG 0.96)の真因は、
  ターゲット外れ値の一律クリップが正規の裾信号を破壊していたこと。対策として <strong>Y外れ値クリップを
  「する/しない」の交差検証選択に変更</strong>(train_bridge.py <code>_select_y_winsorize_cv</code>)。
  さらに第2弾として、AGのL1→L2スタッキングに相当するマージン(真因: スタッキング/バギング不在)へ
  <strong>LightGBM foldバギング</strong>(追加学習コストほぼゼロで分散を下げる。既存blend形式のまま
  native/JSエンジンは無改修)を追加。実測で <code>pub_parkinsons</code> +0.026・<code>real_diabetes</code>
  +0.014 の改善を確認した一方、<code>categorical_high</code> で-0.026の回帰も観測(GP/MLPの内在的な
  乱数変動が疑われるが完全な切り分けはできていない)。1-SEルール・重複行グループCVは実装・実測の結果
  複数データセットで回帰(最大-0.21)を招いたため撤回済み。
  第3弾(2026-07-27)では、datetime列がID列扱いで破棄され未回収だった信号(真因④)へ対応する
  <strong>datetime_parts</strong>を<code>.treg</code> v6拡張として実装(hour/dow/month/epoch_daysの
  4派生列に展開、3予測エンジン全てに移植・パリティ検証済み)。<code>pub_appliances</code>で
  thorough test R² 0.1997→0.2493(+0.0496)・quick 0.1249→0.1917(+0.0668)を実測、date列除外の
  警告も解消。40問全体では他39問がほぼ完全に横ばい(平均thorough R² 0.7699→0.7702)で、副作用は
  実質ゼロと確認。
  第4弾(2026-07-30)では、AG負け問題の再度のなぜなぜ分析で判明した<code>pub_parkinsons</code>の
  反復測定構造(31被験者がtrain/testに100%重複)に対応する<strong>合成キーtarget encoding</strong>
  (低カーディナリティ数値列の組み合わせ、例: age+sexを暗黙の被験者IDとして検出しtarget
  encoding化)を<code>.treg</code> v7拡張として実装。実測で<code>pub_parkinsons</code> thorough
  test R² 0.9016→<strong>0.9666</strong>(+0.065、AGの0.961を上回る水準に到達)を確認、40問全体では
  回帰0件・改善2件(pub_parkinsons以外はcategorical_high +0.024、GP/Blendの既知の乱数変動域内)。
  同時に検証した「dedup-CVゲート」(重複行によるBlend重みfitの汚染対策)は実測で改善が確認できず
  撤回(real_winequalityで-0.0025、pub_concreteで測定不能)。
  <strong>AG側の数値は2026-07-26時点のまま(未再実行、参考値)</strong>: Stage1のLGBM foldバギングで
  treg thoroughの学習時間が伸びており、本来はAGのtime_limitも合わせて再実行すべきだが、
  コスト対効果を鑑み今回は見送った。treg側の実測改善(特にpub_parkinsons)の解釈には影響しないが、
  勝敗数の集計はAG側がやや不利な(短い)時間バジェットのままである点に留意。
  第5弾(2026-07-30)では、AG勝ち10問のなぜなぜ分析で<code>pub_automgp</code>がStage3で新規に
  AG勝ちへ転落していたことが判明(0.916→0.907、-0.009)。原因は合成キー検出が
  <code>cylinders</code>/<code>model_year</code>/<code>origin</code>という3つの独立した意味を持つ
  低カーディナリティ数値列を偶然すべて拾い、73グループの疑似キーを作っていたこと
  (pub_parkinsonsのage+sexのような真の反復測定構造ではない)。対策として合成キー候補の列数上限を
  2列に制限(<code>NUMERIC_KEY_MAX_COLS</code>、pub_parkinsonsはage+sexの2列のみのため無影響)。
  実測で<code>pub_automgp</code> 0.907→0.916に回復(tieへ復帰)。pub_parkinsonsは候補列数が
  2列(age+sex)のみで本対策の影響を受けないことを再学習で確認済み(thorough test R² 0.9645、
  乱数変動範囲内でAGとほぼ並ぶtieに復帰)。</p>
</header>

<h2>サマリー</h2>
{summary_tiles(summ)}

<h2>データセット別 勝敗(AG − treg の test R² 差)</h2>
<div class="card">{svg_gap(rows)}</div>
<p class="note">中央=互角。<span class="hl-treg">青(左)</span>は treg 優位、<span class="hl-ag">橙(右)</span>は AG 優位。
バーの長さが R² 差の大きさ。同一 test 集合・同一採点なので絶対比較。</p>

<h2>treg vs AG 散布図</h2>
<div class="card">{svg_scatter(rows)}</div>
<p class="note">対角線より<strong>上</strong>にある点は AG 優位、<strong>下</strong>は treg 優位。
右上に密集するほど両者とも高精度(易しい問題)。左下は両者とも苦戦(ノイズ支配・小標本など)。</p>

<h2>全問結果(test 集合・中立採点)</h2>
<div class="card">{table(rows)}</div>

<footer>採点は純numpy(R²=1−SS_res/SS_tot)。treg の自己申告R²(OOF/val)は比較に不使用。
AutoGluon 1.5.0 / preset=good_quality / Windows安定化(Ray不使用・dynamic_stacking無効)。
分割シード=42・test=25%。生成: _ag_benchmark/build_dashboard.py</footer>
</div>"""

    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>T-regressor vs AutoGluon — 30問ベンチマーク</title>
<style>{css}</style></head><body>{body}</body></html>"""


def main():
    with open(VIZ, encoding="utf-8") as f:
        data = json.load(f)
    html_out = build(data["rows"], data["summary"])
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(html_out)
    # Artifact 公開用(head/body スケルトンで自動ラップされるため、style + body中身のみ)
    style = html_out.split("<style>", 1)[1].split("</style>", 1)[0]
    body_inner = html_out.split("<body>", 1)[1].rsplit("</body>", 1)[0]
    art = os.path.join(bc.RESULTS_DIR, "dashboard_body.html")
    with open(art, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"<style>{style}</style>\n{body_inner}")
    print(f"生成: {OUT} と {art}  ({len(data['rows'])}問, {len(html_out)} bytes)")


if __name__ == "__main__":
    main()
