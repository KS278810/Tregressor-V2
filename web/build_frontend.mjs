// web/index.html と web/offline.html を、単一の共通フロントエンド ../frontend/index.html
// (Tauri/exe版と共有)から生成するビルドスクリプト。
//
// 背景: 以前は exe版(frontend/index.html)とWeb版(web/index.html・offline.html)を
// 別々のHTML/CSS/JSとして手で二重メンテしていたため、Web版のGUIがexe版より簡素で見た目が
// 揃わなくなっていた。frontend/index.html 側に IS_TAURI 分岐の Platform 抽象化層を実装し、
// バックエンド呼び出し(Tauri invoke/event ↔ Pyodide直接呼び出し)だけを切り替えられるようにした
// ことで、見た目・演出・UIロジックは完全に単一ソースで共有できる。
// このスクリプトが行うのはアセットパスの置換など「配布形態ごとに機械的に決まる差分」のみ。
//
// 実行: cd web && node build_frontend.mjs
// (frontend/index.html を編集したら、Web版を最新化するために必ず再実行すること)
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const WEB_DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.dirname(WEB_DIR);
// 通常は ../frontend/index.html を読む。第1引数で上書き可能(CI/検証用)。
const SRC = process.argv[2] ? path.resolve(process.argv[2]) : path.join(ROOT, "frontend", "index.html");

const GENERATED_NOTICE =
  "<!-- 自動生成ファイル。直接編集しないこと。\n" +
  "     生成元: frontend/index.html (exe版と共通のフロントエンド)\n" +
  "     生成コマンド: cd web && node build_frontend.mjs -->\n";

// ③ SIMULATE(What-ifスライダー)は Web版だけの機能で、学習直後に手元にある .treg を
// JS推論エンジンに渡して1行予測をリアルタイムに回す。エンジンは既にパリティ検証済みの
// js_predict_poc/predict-core.js があるため、**それを1行も変更せずそのまま埋め込む**
// (predict_template.html のように内容を書き写すと5個目の複製になり、CLAUDE.mdルール7の
//  管理対象が増えてしまう)。frontend/index.html 側にはプレースホルダだけを置き、
// ここで機械的に差し込む＝「配布形態ごとに機械的に決まる差分」の範疇に収める。
const PREDICT_CORE_PLACEHOLDER = "/* __PREDICT_CORE_INLINE__ */";

function loadPredictCore() {
  const p = path.join(WEB_DIR, "js_predict_poc", "predict-core.js");
  if (!fs.existsSync(p)) {
    throw new Error(`predict-core.js が見つかりません: ${p}`);
  }
  let src = fs.readFileSync(p, "utf-8");
  // Node向けの CommonJS エクスポートだけをブラウザ用のグローバル公開に読み替える。
  // (ロジック部分には一切触れない。差し替えるのはこの1行のみ)
  const exportRe = /module\.exports\s*=\s*\{[^}]*\};?/;
  if (!exportRe.test(src)) {
    throw new Error("predict-core.js の module.exports が見つかりません（想定外の形式）");
  }
  src = src.replace(exportRe,
    "window.TregPredictCore = { loadTreg, predictRow, roundHalfAwayFromZero };");
  return "// ─── 埋め込み: web/js_predict_poc/predict-core.js（自動挿入・編集禁止） ───\n"
       + "(function(){\n" + src + "\n})();";
}

function loadSource() {
  let html = fs.readFileSync(SRC, "utf-8");
  // exe版は Tauri の treg:// カスタムスキーム経由で reference/ 配下の画像を配信する。
  // Web版は静的ファイルとして assets/ 配下から配信するため、パスだけ置換する。
  const refCount = (html.match(/reference\//g) || []).length;
  html = html.replaceAll("reference/", "assets/");

  if (!html.includes(PREDICT_CORE_PLACEHOLDER)) {
    throw new Error(
      `${PREDICT_CORE_PLACEHOLDER} が frontend/index.html に見つかりません。` +
      "SIMULATE の推論エンジン挿入位置が失われています。");
  }
  const core = loadPredictCore();
  html = html.replace(PREDICT_CORE_PLACEHOLDER, () => core);
  return { html, refCount, coreBytes: core.length };
}

function syncAssets() {
  // frontend/reference/ が原本のアセット一式。web/assets/ に無いものだけ補って同期する。
  const refDir = path.join(ROOT, "frontend", "reference");
  const assetsDir = path.join(WEB_DIR, "assets");
  fs.mkdirSync(assetsDir, { recursive: true });
  const needed = ["logo2.png", "robo2_ok.gif", "robo2_training.gif", "robo2_completed.gif", "icon256.png"];
  const copied = [];
  for (const f of needed) {
    const srcPath = path.join(refDir, f);
    const dstPath = path.join(assetsDir, f);
    // 低-M21: 以前は「web/assets/ に無い場合だけコピー」だったため、
    // frontend/reference/ 側で画像を差し替えても(ファイル名を変えない限り)
    // web版には反映されずビルドが古いアセットのまま「成功」してしまっていた。
    // アセットは軽量なのでビルドの度に常に上書きコピーする。
    if (fs.existsSync(srcPath)) {
      fs.copyFileSync(srcPath, dstPath);
      copied.push(f);
    }
  }

  // 「学習済モデルのDL」がブラウザ上で単体HTMLを組み立てるためのベーステンプレート。
  // HTTP版はこのファイルを fetch("./predict_template.html") で取得する
  // (オフライン版は offline-embed.js に文字列同梱済み。build_offline.mjs 参照)。
  // predict_template.html は web/ 直下に直接置かれている(生成元は無い、直接編集)ため
  // コピーは不要。存在確認だけ行う。
  const predictTemplatePath = path.join(WEB_DIR, "predict_template.html");
  if (!fs.existsSync(predictTemplatePath)) {
    console.warn("警告: web/predict_template.html が見つかりません。「学習済モデルのDL」が動作しません。");
  }

  return copied;
}

// frontend/index.html は CRLF 改行のため、置換パターンは \r? を許容しておく。
const DOCTYPE_RE = /<!DOCTYPE html>\r?\n/;
const CHARSET_RE = /<meta charset="UTF-8">/;

function buildHttpVariant(html) {
  let out = html;
  out = out.replace("<title>T-regressor</title>", "<title>T-regressor Web</title>");
  out = out.replace(CHARSET_RE, (m) => m + "\n    <link rel=\"icon\" href=\"./assets/icon256.png\">");
  out = out.replace(DOCTYPE_RE, (m) => m + GENERATED_NOTICE);
  return out;
}

function buildOfflineVariant(html) {
  let out = html;
  out = out.replace("<title>T-regressor</title>", "<title>T-regressor Web（オフライン）</title>");
  out = out.replace(CHARSET_RE, (m) => m + "\n    <link rel=\"icon\" href=\"./assets/icon256.png\">");
  out = out.replace(DOCTYPE_RE, (m) => m + GENERATED_NOTICE);
  // offline-engine.js は Import Maps を document.write で同期注入する必要があるため、
  // 本編スクリプトより前・pyodide.js の読込より前に置く(詳細は offline-engine.js 冒頭コメント参照)。
  const inject =
    '<!-- file:// で直接ダブルクリックして開ける版。ESモジュール/fetch は file:// でCORSブロックされるため\n' +
    '     使わず、全データ(Pyodide本体+依存ライブラリ+Pythonソース+サンプルCSV)を offline-embed.js に\n' +
    '     同梱し、offline-engine.js が Import Maps + fetchオーバーライドでネットワーク非依存に読み込む。 -->\n' +
    '<script src="./offline-embed.js"></script>\n' +
    '<script src="./offline-engine.js"></script>\n' +
    '<script src="./vendor/pyodide/pyodide.js"></script>\n';
  if (!out.includes("<script>")) throw new Error("main <script> block not found in source");
  out = out.replace("<script>", inject + "<script>");
  return out;
}

function main() {
  const { html, refCount, coreBytes } = loadSource();
  console.log(`読込: ${SRC} (reference/ -> assets/ 置換 ${refCount}箇所)`);
  console.log(`埋込: js_predict_poc/predict-core.js (${(coreBytes / 1024).toFixed(1)} KB) → SIMULATE 用推論エンジン`);

  const copied = syncAssets();
  if (copied.length) console.log(`アセット同期(常に上書き): ${copied.join(", ")}`);

  const httpOut = path.join(WEB_DIR, "index.html");
  fs.writeFileSync(httpOut, buildHttpVariant(html));
  console.log(`生成: ${httpOut}`);

  const offlineOut = path.join(WEB_DIR, "offline.html");
  fs.writeFileSync(offlineOut, buildOfflineVariant(html));
  console.log(`生成: ${offlineOut}`);

  console.log("完了。offline-embed.js が古い場合は続けて node build_offline.mjs を実行すること。");
}

main();
