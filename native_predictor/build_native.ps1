# native_predictor/build_native.ps1
# predict_native_v2.cpp を MinGW (g++) でネイティブ予測 EXE にビルドする。
# 外部依存ゼロ・静的リンクなので配布先に MinGW/ランタイムのインストールは不要。
#
# 前提: MinGW (g++ / windres) が PATH にあるか、下記 $MinGwBin で指定した場所にあること
# 使い方: .\build_native.ps1 [-MinGwBin "C:\MinGW\bin"]

param([string]$MinGwBin = "C:\MinGW\bin")

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot

$GxxExe    = Join-Path $MinGwBin "g++.exe"
$WindresExe = Join-Path $MinGwBin "windres.exe"
if (-not (Test-Path $GxxExe))    { throw "g++.exe が見つかりません: $GxxExe（MinGW のパスを -MinGwBin で指定してください）" }
if (-not (Test-Path $WindresExe)) { throw "windres.exe が見つかりません: $WindresExe" }

$Src     = Join-Path $ScriptDir "predict_native_v2.cpp"
$RcFile  = Join-Path $ScriptDir "predict_native.rc"
$ResObj  = Join-Path $ScriptDir "predict_native_res.o"
$OutExe  = Join-Path $ScriptDir "predict_native.exe"

Write-Host "[1/2] アイコンリソースをコンパイル中 (windres)..." -ForegroundColor Cyan
& $WindresExe $RcFile -O coff -o $ResObj
if ($LASTEXITCODE -ne 0) { throw "windres 失敗" }

Write-Host "[2/2] ビルド中 (g++)..." -ForegroundColor Cyan
# -std=gnu++17: 中-M2で使う _wfopen 等のMinGW非標準拡張(stdio.h上で__STRICT_ANSI__に
#   よってガードされている)を見えるようにするため、厳格ANSIモードを敷く -std=c++17
#   ではなく gnu++17 を使う(言語機能自体はC++17のまま、CRT拡張のみ追加で見える化)。
# -lshell32: CommandLineToArgvW(cp932外の文字を含むCSVパスのD&D対応、中-M2)に必要。
& $GxxExe -O2 -std=gnu++17 -static -mwindows -s $Src $ResObj -lshell32 -o $OutExe
if ($LASTEXITCODE -ne 0) { throw "g++ ビルド失敗" }

Remove-Item $ResObj -ErrorAction SilentlyContinue

$sizeKB = [math]::Round((Get-Item $OutExe).Length / 1KB, 1)
Write-Host ""
Write-Host "ビルド成功: $OutExe (${sizeKB} KB)" -ForegroundColor Green
Write-Host "改修後は build_portable.ps1 を実行して配布フォルダに反映してください。" -ForegroundColor Yellow
