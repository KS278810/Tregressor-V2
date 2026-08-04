# deploy_public.ps1 — Web版を成果物専用の公開リポジトリ(KS278810/T-regressor-app)へ
# 手動で反映するためのスクリプト。ローカルで実行する(GitHub Actions・デプロイ鍵・
# Secretsは使わない。ユーザーが既に持っている git 認証情報でそのままpushする)。
#
# 【背景】Tregressor-V2(本体・ソース一式)を非公開化する一方、Web版アプリは
# 今まで通り無料で公開したいため、「成果物(ビルド済みの静的ファイル)だけを配る」
# 公開リポジトリを別に用意した。ソースは非公開のTregressor-V2側にのみ残る。
#
# 【やっていること】
#   1. frontend/index.html から web/index.html 等を作り直す(build_frontend.mjs)
#   2. アプリ本体ロジック(JS)とpy/配下のPythonソースを難読化する(build_obfuscated.mjs --site)
#      ※クライアントサイド実行である以上、真の秘匿はできない点はweb/README.md参照
#   3. 配信に必要なファイルだけを web/_public_site/ に集める
#   4. web/_public_site/ を、KS278810/T-regressor-app の main ブランチへ push する
#      (初回はこのフォルダをgit初期化してremoteを設定、以後は同じフォルダを使い回す)
#
# 【前提】
#   - Node.js, Python3 + python-minifier(`pip install python-minifier`) が導入済みであること
#   - `cd web && npm install` を一度実行済みであること(terser/javascript-obfuscator)
#   - あなたのPCで既に Tregressor-V2 に git push できている = その認証情報が
#     T-regressor-app にもそのまま使える(同じGitHubアカウントの別リポジトリのため)
#
# 【使い方】Web版を更新したいとき、このスクリプトを実行するだけ:
#   cd web
#   .\deploy_public.ps1
#
# 初回のみ、実行後に T-regressor-app の Settings → Pages → Source を
# 「Deploy from a branch: main / (root)」に設定すること(ブランチができてから選択可能)。

$ErrorActionPreference = "Stop"

$PUBLIC_REPO_URL = "https://github.com/KS278810/T-regressor-app.git"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$SITE_DIR = Join-Path $SCRIPT_DIR "_public_site"

Write-Host "=== 1. 生成物を作り直す(frontend/index.html が唯一のソース) ===" -ForegroundColor Cyan
Push-Location $SCRIPT_DIR
node build_frontend.mjs
if ($LASTEXITCODE -ne 0) { throw "build_frontend.mjs が失敗しました" }

Write-Host "=== 2. 難読化版を生成(--site: Pages配信対象のみ) ===" -ForegroundColor Cyan
node build_obfuscated.mjs --site
if ($LASTEXITCODE -ne 0) { throw "build_obfuscated.mjs が失敗しました" }

Write-Host "=== 3. 配信するファイルだけを web/_public_site/ に集める ===" -ForegroundColor Cyan
if (Test-Path $SITE_DIR) {
    # .git だけは残して中身を入れ替える(初回以降はgit履歴を維持するため)
    Get-ChildItem $SITE_DIR -Force | Where-Object { $_.Name -ne ".git" } | Remove-Item -Recurse -Force
} else {
    New-Item -ItemType Directory -Path $SITE_DIR | Out-Null
}

Copy-Item "dist_obfuscated\index.html"            $SITE_DIR
Copy-Item "dist_obfuscated\simulate_template.html" $SITE_DIR
Copy-Item "dist_obfuscated\predict_template.html"  $SITE_DIR
Copy-Item "dist_obfuscated\treg-engine.js"         $SITE_DIR
Copy-Item "dist_obfuscated\treg-worker.js"         $SITE_DIR
Copy-Item "dist_obfuscated\treg-worker-client.js"  $SITE_DIR
Copy-Item "dist_obfuscated\sample_data.csv"        $SITE_DIR
Copy-Item -Recurse "dist_obfuscated\assets"        (Join-Path $SITE_DIR "assets")
Copy-Item -Recurse "dist_obfuscated\py"            (Join-Path $SITE_DIR "py")
Copy-Item -Recurse "vendor"                        (Join-Path $SITE_DIR "vendor")

New-Item -ItemType File -Path (Join-Path $SITE_DIR ".nojekyll") -Force | Out-Null
$stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
"deployed: $stamp" | Out-File -Encoding utf8 (Join-Path $SITE_DIR ".build-stamp")

$sizeMB = "{0:N1}" -f ((Get-ChildItem $SITE_DIR -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB)
Write-Host "配信物サイズ: $sizeMB MB"

Write-Host "=== 4. T-regressor-app へ push ===" -ForegroundColor Cyan
Push-Location $SITE_DIR
try {
    if (-not (Test-Path ".git")) {
        git init -q
        git checkout -q -b main
        git remote add origin $PUBLIC_REPO_URL
    }
    git add -A
    $hasChanges = (git status --porcelain)
    if (-not $hasChanges) {
        Write-Host "変更なし。pushをスキップします。" -ForegroundColor Yellow
    } else {
        git commit -q -m "deploy: $stamp"
        git push -f origin main
        Write-Host "`n完了: https://ks278810.github.io/T-regressor-app/ に反映されます(数分かかることがあります)" -ForegroundColor Green
    }
} finally {
    Pop-Location
}
Pop-Location
