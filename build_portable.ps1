# build_portable.ps1 — T-regressor ポータブル配布物を再組み立て (Tauri V2)
#
# 実行方法:
#   cd D:\_claude\15_TregV2
#   .\build_portable.ps1
#
# 前提条件:
#   - Rust / cargo が PATH にある
#   - cargo-tauri (tauri-cli) がインストール済み
#     （未導入の場合: cargo install tauri-cli --version "^2"）
#   - dist_portable\T-regressor\python-embed\ が既にセットアップ済み
#     (numpy, pandas, lightgbm, scipy が入っていること。scikit-learnは不要)
#
# python-embed の自動セットアップ手順（High A-1: 依存ピン留め）:
#   このスクリプト自体はpython-embedの初回セットアップを行わない（[1/4]は確認のみ）。
#   未セットアップの場合は以下の手順で用意すること。バージョンは requirements-embed.txt
#   に固定してあり、再現可能なビルドのために毎回このファイルを使うこと。
#
#   1. Python 3.11 embeddable package (64-bit, win_amd64) を取得し、
#      dist_portable\T-regressor\python-embed\ に展開する。
#      取得元: https://www.python.org/downloads/windows/ の
#        "Windows embeddable package (64-bit)" (例: python-3.11.9-embed-amd64.zip)。
#      現物のビルド検証は Python 3.11.15 で行った(python.exe --version で確認可能)。
#      ダウンロードしたzipのSHA256は、python.orgの当該リリースページ
#      (例 https://www.python.org/downloads/release/python-3119/) に掲載された
#      チェックサムと必ず突き合わせること（zipはリリースごとにハッシュが変わるため
#      固定値をこのファイルに埋め込むと将来のバージョン更新時に検証が形骸化する。
#      取得の都度、公式ページのハッシュと比較する運用とする）。
#   2. pip はembeddable packageには同梱されないため、get-pip.py で個別導入する:
#        Invoke-WebRequest https://bootstrap.pypa.io/get-pip.py -OutFile get-pip.py
#        & dist_portable\T-regressor\python-embed\python.exe get-pip.py
#      加えて python311._pth の `#import site` 行のコメントを外し、
#      site-packages（--target先）が認識されるようにしておく。
#   3. requirements-embed.txt を使って依存を固定導入する:
#        & dist_portable\T-regressor\python-embed\python.exe -m pip install `
#            -r requirements-embed.txt `
#            --target dist_portable\T-regressor\python-embed\Lib\site-packages
#      （scipy は lightgbm の依存で自動的に入るが [4/4]ではなく prune_embed.ps1 の
#        pruning で軽量 stub に置換される。scikit-learn は不要 —
#        _light.py の自前実装で置換済みのため requirements-embed.txt にも含めない）
#   詳細・依存の全リストは requirements-embed.txt を参照。
#
# 出力:
#   dist_portable/T-regressor/  (デプロイ済みフォルダを上書き更新)
#
# 配布物の構成:
#   T-regressor/
#     T-regressor.exe        <- Tauri + Rust ビルド成果物
#     index.html             <- フロントエンド
#     train_bridge.py        <- 学習スクリプト
#     predict_template.py    <- 予測スクリプト
#     bat_template.txt       <- bat テンプレート
#     reference/             <- UI 画像 (GIF/PNG)
#     python-embed/          <- Python 実行環境 (依存なし自己完結)
#     native_dist/           <- ネイティブ予測 EXE (predict_native.exe)

$ErrorActionPreference = "Stop"

$Root        = Split-Path -Parent $MyInvocation.MyCommand.Path
$DistRoot    = Join-Path $Root "dist_portable"
$AppName     = "T-regressor"
$Dist        = Join-Path $DistRoot $AppName
$PythonEmbed = Join-Path $Dist "python-embed"

function Write-Step([int]$n, [int]$total, [string]$msg) {
    Write-Host ""
    Write-Host "[$n/$total] $msg" -ForegroundColor Cyan
}

# ════════════════════════════════════════════════════════════
# [1/4] python-embed 確認
# ════════════════════════════════════════════════════════════
Write-Step 1 4 "python-embed 確認"

$pythonExe = Join-Path $PythonEmbed "python.exe"
$lgbmDir   = Join-Path $PythonEmbed "Lib\site-packages\lightgbm"

if (-not (Test-Path $pythonExe)) {
    throw @"
dist_portable\T-regressor\python-embed\python.exe が見つかりません。
python-embed を手動でセットアップしてください（詳細・SHA256確認手順はこのファイル冒頭のコメント参照）:
  1. https://www.python.org/downloads/windows/ から embeddable package (64-bit, Python 3.11) を取得
     し、ダウンロードしたzipのSHA256をpython.orgのリリースページ記載値と照合する
  2. dist_portable\T-regressor\python-embed\ に解凍
  3. get-pip.py で pip を導入し、python311._pth の `#import site` のコメントを外す
  4. requirements-embed.txt で依存を固定導入:
       & "$pythonExe" -m pip install -r requirements-embed.txt --target `
           dist_portable\T-regressor\python-embed\Lib\site-packages
     （scipy は lightgbm の依存で自動的に入るが prune_embed.ps1 が軽量 stub に置換する。
       scikit-learn は不要 — _light.py の自前実装で置換済みのため導入しない）
"@
}

if (-not (Test-Path $lgbmDir)) {
    throw @"
python-embed に lightgbm が見つかりません。
requirements-embed.txt を使って固定版を導入してください:
  & "$pythonExe" -m pip install -r requirements-embed.txt --target `
      dist_portable\T-regressor\python-embed\Lib\site-packages
  （scikit-learn は不要。scipy は依存で入るが pruning で stub 化される）
"@
}

Write-Host "     python-embed 確認済み" -ForegroundColor Green
# sklearn は _light.py の自前実装で置換済みのため検証しない（[4/4] pruning で除去される）。
# scipy は stub でも import 可能。実行に必須なのは lightgbm/numpy/pandas。
$testResult = & $pythonExe -c "import lightgbm, numpy, pandas, scipy; print('OK')" 2>&1
if ($testResult -notmatch "OK") {
    throw "インポート検証に失敗しました:`n$testResult"
}
Write-Host "     必須パッケージ インポート: OK (lightgbm/numpy/pandas/scipy)" -ForegroundColor Green

# ════════════════════════════════════════════════════════════
# [2/4] embed Python の pruning（不要ファイル削除で配布容量を削減）
# ════════════════════════════════════════════════════════════
# cargo tauri build（build.rs が python-embed を zip 化して EXE に埋め込む）より前に
# 実行する。以前は cargo tauri build の後段でpruningしていたため、pruning未実施の
# python-embed（scipy本体・sklearn等を含む肥大化した状態）を初回ビルド時にそのまま
# zip化・EXE埋め込みしてしまう可能性があった(中-13)。pruning は dist_portable\...\python-embed
# に対して完結する処理で、cargo tauri build の出力には依存しないため安全に前倒しできる。
Write-Step 2 4 "embed Python の pruning"

$pruneScript = Join-Path $Root "prune_embed.ps1"
if (Test-Path $pruneScript) {
    & $pruneScript -PythonEmbed $PythonEmbed
} else {
    Write-Host "     prune_embed.ps1 が見つからないためスキップ" -ForegroundColor Yellow
}

# ════════════════════════════════════════════════════════════
# [3/4] cargo tauri build --no-bundle
# ════════════════════════════════════════════════════════════
Write-Step 3 4 "cargo tauri build --no-bundle"

Push-Location $Root
try {
    $tauriCheck = cargo tauri --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "     tauri-cli が見つかりません。インストール中..." -ForegroundColor Yellow
        cargo install tauri-cli --version "^2"
        if ($LASTEXITCODE -ne 0) { throw "tauri-cli のインストールに失敗しました" }
    } else {
        Write-Host "     tauri-cli: $tauriCheck"
    }

    Write-Host "     ビルド中（初回は数分かかります）..."
    cargo tauri build --no-bundle
    if ($LASTEXITCODE -ne 0) { throw "cargo tauri build に失敗しました" }
} finally {
    Pop-Location
}

$TauriExe = Join-Path $Root "src-tauri\target\release\t-regressor.exe"
if (-not (Test-Path $TauriExe)) { throw "ビルド出力が見つかりません: $TauriExe" }
$exeSizeMB = [math]::Round((Get-Item $TauriExe).Length / 1MB, 1)
Write-Host "     ビルド完了: t-regressor.exe ($exeSizeMB MB)" -ForegroundColor Green

# ════════════════════════════════════════════════════════════
# [4/4] 配布フォルダを更新
# ════════════════════════════════════════════════════════════
Write-Step 4 4 "配布フォルダを更新"

New-Item -ItemType Directory -Force $Dist | Out-Null

# EXE
Copy-Item $TauriExe (Join-Path $Dist "T-regressor.exe") -Force
Write-Host "     T-regressor.exe 更新"

# フロントエンドHTML
Copy-Item (Join-Path $Root "frontend\index.html") (Join-Path $Dist "index.html") -Force
Write-Host "     index.html 更新"

# reference/ (GIF/PNG)
$refDest = Join-Path $Dist "reference"
New-Item -ItemType Directory -Force $refDest | Out-Null
foreach ($f in @("robo2_ok.gif","robo2_training.gif","robo2_completed.gif","logo2.png")) {
    $src = Join-Path $Root "frontend\reference\$f"
    if (Test-Path $src) { Copy-Item $src (Join-Path $refDest $f) -Force }
}
Write-Host "     reference/ 更新"

# Python スクリプト
foreach ($f in @("train_bridge.py","predict_template.py","_light.py","bat_template.txt")) {
    $src = Join-Path $Root $f
    if (Test-Path $src) { Copy-Item $src (Join-Path $Dist $f) -Force; Write-Host "     $f 更新" }
}

# native_dist/ (predict_native.exe)
$nativeSrc  = Join-Path $Root "native_predictor\predict_native.exe"
$nativeDest = Join-Path $Dist "native_dist"
if (Test-Path $nativeSrc) {
    New-Item -ItemType Directory -Force $nativeDest | Out-Null
    Copy-Item $nativeSrc (Join-Path $nativeDest "predict_native.exe") -Force
    Write-Host "     native_dist/ 更新"
}

$distSizeMB = [math]::Round(
    (Get-ChildItem $Dist -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
)

Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host " 更新完了: $Dist" -ForegroundColor Green
Write-Host " 合計サイズ: $distSizeMB MB" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
