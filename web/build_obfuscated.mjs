// build_obfuscated.mjs — index.html / offline.html のアプリ本体ロジック(inline <script>)
// と自作アダプタJS(treg-engine.js等)、および py/ 配下のPythonソース(train_bridge.py等)を
// 難読化した配布物を dist_obfuscated/ に生成する。
//
// リポジトリ直下の index.html / offline.html はあえて「そのまま読める配布物」として
// 追跡している(README参照)。この方針とは矛盾させず、難読化版はビルド成果物として
// dist_obfuscated/ にのみ出力し、コミット対象外(.gitignore)とする。
//
// 【2026-08 追記】GitHub Pages配信(pages.yml)からこの難読化ビルドを呼び出すようになった。
// Pages配信は index.html / simulate_template.html のみを対象とし(offline.html は配信対象外
// のため処理しない)、--site フラグで対象を絞り込める。
//
// 【2026-08 追記】py/ 配下の train_bridge.py / _light.py / predict_template.py は、
// Web版がブラウザ内Pyodideでソースをそのまま実行する設計上、非公開化しても隠せない
// (クライアントに平文で配信される)。カジュアルな無断転用の抑止だけを目的に、
// python-minifier(コメント/docstring除去+識別子短縮。ロジックの意味は変えない、
// tests/によるパリティ検証済み)を通す。真の秘匿ではない点はJS同様の注意書きを参照。
//
// 実行前提: index.html / offline.html / vendor / assets / py 等が最新であること
// (`node build_frontend.mjs` [`node build_offline.mjs`] を先に実行しておくこと)。
// python-minifier が必要(`pip install python-minifier`)。
//
// 実行方法:
//   cd web
//   npm install
//   node build_obfuscated.mjs          # フル版(index.html + offline.html 両方)
//   node build_obfuscated.mjs --site   # Pages配信用(index.html + simulate_template.html のみ)

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { minify } from 'terser';
import JavaScriptObfuscator from 'javascript-obfuscator';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.join(__dirname, 'dist_obfuscated');
const SITE_MODE = process.argv.includes('--site');

const OBFUSCATOR_OPTS = {
    compact: true,
    controlFlowFlattening: false,
    stringArray: true,
    stringArrayThreshold: 0.75,
    stringArrayEncoding: ['base64'],
    renameGlobals: false,
    numbersToExpressions: false,
    splitStrings: false,
};

// 難読化対象: アプリ本体ロジックのみ。vendor/pyodide/(サードパーティ)・
// offline-embed.js(埋め込みデータblob、約59MB)は対象外(壊れる/意味がないため)。
const ADAPTER_JS_FILES_FULL = [
    { name: 'treg-engine.js', module: true },
    { name: 'treg-worker.js', module: true },
    { name: 'treg-worker-client.js', module: true },
    { name: 'offline-engine.js', module: false },
];
// --site モードでは offline版専用の offline-engine.js は不要(Pages配信対象外)。
const ADAPTER_JS_FILES = SITE_MODE
    ? ADAPTER_JS_FILES_FULL.filter((f) => f.name !== 'offline-engine.js')
    : ADAPTER_JS_FILES_FULL;

// index.html は常に対象。simulate_template.html は build_frontend.mjs が index.html の
// 複製として生成する「モデルをDL」用の雛形で、中身(inline scriptを含め)は生成直後は
// index.html と同一のため同じ扱いで難読化する。offline.html は --site モードでは
// 対象外(Pages配信しないため)。
const HTML_FILES_FULL = [
    { name: 'index.html', module: true },
    { name: 'simulate_template.html', module: true },
    { name: 'offline.html', module: false },
];
const HTML_FILES = SITE_MODE
    ? HTML_FILES_FULL.filter((f) => f.name !== 'offline.html')
    : HTML_FILES_FULL;

// py/ 配下で難読化(minify)する対象。predict_template.py はコマンドライン一括推論用の
// 独立実行スクリプトで、train_bridge.py/_light.py と依存関係が無いため個別に処理できる。
const PY_FILES = ['train_bridge.py', '_light.py', 'predict_template.py'];

const PY_LICENSE_BANNER =
    '# Copyright (c) 2026 Kohei Shintani. Licensed under CC BY-NC 4.0\n' +
    '# (Attribution-NonCommercial): https://creativecommons.org/licenses/by-nc/4.0/\n' +
    '# Commercial use requires prior permission (see LICENSE in the source repository).\n' +
    '# This file has been minified for distribution (comments/docstrings removed).\n';

const JS_LICENSE_BANNER =
    '/* Copyright (c) 2026 Kohei Shintani. Licensed under CC BY-NC 4.0 (Attribution-NonCommercial).\n' +
    ' * https://creativecommons.org/licenses/by-nc/4.0/\n' +
    ' * Commercial use requires prior permission (see LICENSE in the source repository).\n' +
    ' * This file has been minified/obfuscated for distribution. */\n';

// --site モードでは predict_template.html(単体HTMLモデル書き出し用テンプレート)は
// __TREG_BASE64__ プレースホルダをJS未実行の生テキスト置換で扱う仕組みのため対象外
// (難読化するとプレースホルダの文字列が変形し「モデルをDL」機能が壊れる恐れがある)。
const PASSTHROUGH_FULL = ['offline-embed.js', 'vendor', 'assets', 'predict_template.html', 'sample_data.csv'];
const PASSTHROUGH = PASSTHROUGH_FULL;

async function obfuscateJs(code, { module }) {
    const minified = await minify(code, {
        module,
        compress: { drop_console: true },
        mangle: { toplevel: true },
        format: { comments: false },
    });
    const result = JavaScriptObfuscator.obfuscate(minified.code, {
        ...OBFUSCATOR_OPTS,
        sourceType: module ? 'module' : 'script',
    });
    return result.getObfuscatedCode();
}

function replaceLastInlineScript(html, obfuscatedCode) {
    const openTag = '<script>';
    const closeTag = '</script>';
    const openIdx = html.lastIndexOf(openTag);
    const closeIdx = html.indexOf(closeTag, openIdx);
    if (openIdx === -1 || closeIdx === -1) {
        throw new Error('inline <script> ブロックが見つかりません');
    }
    return (
        html.slice(0, openIdx + openTag.length) +
        '\n' + obfuscatedCode + '\n' +
        html.slice(closeIdx)
    );
}

// python-minifier(pip)経由でPythonソースを難読化する。ロジックの意味は変えない
// (comment/docstring除去+識別子短縮のみ)。CIではpython-minifier必須とし、
// 万一未導入なら「無言で平文のまま配信」という事故を避けるため明示的にthrowする。
function minifyPython(srcPath) {
    const result = spawnSync('python3', ['-m', 'python_minifier', srcPath], { encoding: 'utf-8' });
    if (result.status !== 0 || !result.stdout) {
        throw new Error(
            `python-minifier の実行に失敗しました(${srcPath})。` +
            `\`pip install python-minifier\` が導入済みか確認してください。\n` +
            `stderr: ${result.stderr}`
        );
    }
    return result.stdout;
}

async function main() {
    fs.mkdirSync(OUT_DIR, { recursive: true });
    console.log(SITE_MODE ? 'モード: --site (Pages配信用サブセット)' : 'モード: フル');

    for (const { name, module } of ADAPTER_JS_FILES) {
        const srcPath = path.join(__dirname, name);
        const code = fs.readFileSync(srcPath, 'utf8');
        const obfuscated = await obfuscateJs(code, { module });
        fs.writeFileSync(path.join(OUT_DIR, name), JS_LICENSE_BANNER + obfuscated);
        console.log(`難読化(JS): ${name}`);
    }

    for (const { name, module } of HTML_FILES) {
        const srcPath = path.join(__dirname, name);
        if (!fs.existsSync(srcPath)) {
            console.warn(`警告: ${name} が見つかりません。スキップします(先にbuild_frontend.mjs等を実行したか確認)`);
            continue;
        }
        const html = fs.readFileSync(srcPath, 'utf8');
        const openTag = '<script>';
        const closeTag = '</script>';
        const openIdx = html.lastIndexOf(openTag);
        const closeIdx = html.indexOf(closeTag, openIdx);
        const inlineCode = html.slice(openIdx + openTag.length, closeIdx);
        const obfuscated = await obfuscateJs(inlineCode, { module });
        const outHtml = replaceLastInlineScript(html, obfuscated);
        fs.writeFileSync(path.join(OUT_DIR, name), outHtml);
        console.log(`難読化(HTML): ${name}`);
    }

    // py/ 配下のPythonソースをminifyして dist_obfuscated/py/ に出力する。
    const pyOutDir = path.join(OUT_DIR, 'py');
    fs.mkdirSync(pyOutDir, { recursive: true });
    for (const name of PY_FILES) {
        const srcPath = path.join(__dirname, 'py', name);
        if (!fs.existsSync(srcPath)) {
            throw new Error(`${srcPath} が見つかりません`);
        }
        const minified = minifyPython(srcPath);
        fs.writeFileSync(path.join(pyOutDir, name), PY_LICENSE_BANNER + minified);
        console.log(`難読化(Python): py/${name} (${fs.statSync(srcPath).size} → ${minified.length} bytes)`);
    }

    // 残りのアセット(vendor/pyodide, offline-embed.js, predict_template.html, assets等)は
    // 無改変でそのままコピーする(難読化対象外・壊れるため)。
    for (const entry of PASSTHROUGH) {
        const srcPath = path.join(__dirname, entry);
        if (!fs.existsSync(srcPath)) continue;
        fs.cpSync(srcPath, path.join(OUT_DIR, entry), { recursive: true });
    }
    // sample_data.csv・predict_template.htmlはPages配信にも必要。
    fs.mkdirSync(path.join(OUT_DIR), { recursive: true });

    console.log(`\n完了: ${path.relative(process.cwd(), OUT_DIR)}/ に難読化配布物を生成しました。`);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
