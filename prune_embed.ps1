# prune_embed.ps1 — 同梱(embed) Python から不要ファイルを削除（冪等・安全）
#
# 【重要】対象は T-regressor 同梱の embed Python のみ。
#   システム/ローカルの Python・venv には絶対に触れない（下の安全ガード参照）。
#
# 対象は「学習・予測の実行に一切使われないファイル」のみ:
#   - Scripts/           : pip/f2py 等の実行ファイル（import 対象外）
#   - tcl/ + tk DLL      : Tkinter 一式（GUI は Tauri なので不要）
#   - **/tests, **/test  : 各パッケージのテストコード（import 対象外）
#   - __pycache__        : .pyc キャッシュ（実行時に再生成される）
#   - _*test*.pyd        : CPython 自己テスト用モジュール
#
# 単独実行も可:  .\prune_embed.ps1
# pruning 後は必ず tests\verify_rebuild.py と tests\test_harness.py を通し、
# 動的 import 漏れ（実行時 import エラー）がないことを確認すること。

param([string]$PythonEmbed)
$ErrorActionPreference = "Stop"

if (-not $PythonEmbed) {
    $root = Split-Path -Parent $MyInvocation.MyCommand.Path
    $PythonEmbed = Join-Path $root "dist_portable\T-regressor\python-embed"
}

# ── 安全ガード: 想定した embed パス以外は絶対に処理しない ───────────────────
$full = [System.IO.Path]::GetFullPath($PythonEmbed)
if ($full -notmatch 'dist_portable' -or $full -notmatch 'python-embed') {
    throw "安全ガード: 対象パスが embed Python ではありません（dist_portable\...\python-embed 以外）。中止します。`n  対象: $full"
}
if (-not (Test-Path $PythonEmbed)) { throw "python-embed が見つかりません: $full" }
# 同梱の証拠（python.exe が直下にある embeddable 構成）を確認
if (-not (Test-Path (Join-Path $PythonEmbed "python.exe"))) {
    throw "安全ガード: $full 直下に python.exe がありません。embed 環境ではない可能性。中止します。"
}
Write-Host "     対象(embedのみ): $full" -ForegroundColor DarkGray

$sp = Join-Path $PythonEmbed "Lib\site-packages"

function Get-DirSizeMB($p) {
    if (Test-Path $p) {
        [math]::Round((Get-ChildItem $p -Recurse -File -ErrorAction SilentlyContinue |
            Measure-Object Length -Sum).Sum / 1MB, 1)
    } else { 0 }
}

$before = Get-DirSizeMB $PythonEmbed

# 1. Scripts（pip/f2py 等の実行ファイル。実行時 import されない）
$scripts = Join-Path $PythonEmbed "Scripts"
if (Test-Path $scripts) { Remove-Item $scripts -Recurse -Force }

# 2. Tkinter 一式（GUI は Tauri。完全に不要）
$tcl = Join-Path $PythonEmbed "tcl"
if (Test-Path $tcl) { Remove-Item $tcl -Recurse -Force }
foreach ($d in @("tcl86t.dll", "tk86t.dll", "_tkinter.pyd")) {
    $p = Join-Path $PythonEmbed "DLLs\$d"
    if (Test-Path $p) { Remove-Item $p -Force }
}

# 3. CPython 自己テスト用 pyd（_ctypes_test / _testcapi 等）
$dllsDir = Join-Path $PythonEmbed "DLLs"
if (Test-Path $dllsDir) {
    Get-ChildItem $dllsDir -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^_.*test.*\.pyd$' } | Remove-Item -Force
}

# 4. 各パッケージの tests/test ディレクトリ（import 対象外）
if (Test-Path $sp) {
    Get-ChildItem $sp -Recurse -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -in @('tests', 'test') } |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

# 5. __pycache__（.pyc は実行時に再生成される）
Get-ChildItem $PythonEmbed -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# 6. scikit-learn を除去 + scipy を軽量スタブに置換（依存スリム化の本丸）
#    学習・予測は _light.py の自前実装（numpy のみ）を使うため sklearn は不要。
#    lightgbm は dense 専用運用なので scipy 本体（約105MB）は不要で、
#    lightgbm/basic.py が無条件 import する scipy.sparse のクラス定義だけを stub で提供する。
#    冪等: 既に stub 化済みなら何もしない。full scipy/sklearn が混入していれば毎回スリム化。
if (Test-Path $sp) {
    # 6a. sklearn 本体 + dist-info を除去
    $sklearnDir = Join-Path $sp "sklearn"
    if (Test-Path $sklearnDir) { Remove-Item $sklearnDir -Recurse -Force }
    Get-ChildItem $sp -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^(scikit_learn|sklearn)-.*\.dist-info$' } |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    # 6b. scipy が「本物」なら削除して stub 生成（stub は先頭のマーカーで判定）
    $stubMarker = "0.0.0-stub-treg"
    $scipyDir   = Join-Path $sp "scipy"
    $scipyLibs  = Join-Path $sp "scipy.libs"
    $initPy     = Join-Path $scipyDir "__init__.py"
    $isStub     = $false
    if (Test-Path $initPy) {
        $head = Get-Content $initPy -TotalCount 6 -ErrorAction SilentlyContinue
        if ($head -match [regex]::Escape($stubMarker)) { $isStub = $true }
    }
    if (-not $isStub) {
        if (Test-Path $scipyDir)  { Remove-Item $scipyDir  -Recurse -Force }
        if (Test-Path $scipyLibs) { Remove-Item $scipyLibs -Recurse -Force }
        Get-ChildItem $sp -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^scipy-.*\.dist-info$' } |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

        New-Item -ItemType Directory -Force (Join-Path $scipyDir "sparse") | Out-Null
        Set-Content -Path $initPy -Encoding UTF8 -Value @'
# Minimal scipy stub — 依存スリム化のため実 scipy(約105MB) は同梱しない。
# LightGBM が参照する scipy.sparse のクラス/関数だけを提供する。
# 実データは numpy 密行列のみ扱うため、スパース経路は実行時に通らない。
__version__ = "0.0.0-stub-treg"
from . import sparse  # noqa: F401
'@
        Set-Content -Path (Join-Path $scipyDir "sparse\__init__.py") -Encoding UTF8 -Value @'
"""LightGBM dense-only 用の scipy.sparse スタブ。
実データ（numpy 密行列）ではスパース経路に入らないため、isinstance 判定用の
クラスと、万一呼ばれた際に明示エラーを出す関数だけを定義する。"""


class spmatrix:  # 基底クラス（密行列は決してこのインスタンスにならない）
    pass


class csr_matrix(spmatrix):
    def __init__(self, *a, **k):
        raise RuntimeError("scipy.sparse は同梱されていません（dense 専用ビルド）")


class csc_matrix(spmatrix):
    def __init__(self, *a, **k):
        raise RuntimeError("scipy.sparse は同梱されていません（dense 専用ビルド）")


def hstack(*a, **k):
    raise RuntimeError("scipy.sparse.hstack は同梱されていません（dense 専用ビルド）")


def vstack(*a, **k):
    raise RuntimeError("scipy.sparse.vstack は同梱されていません（dense 専用ビルド）")


def issparse(x):
    return False


def isspmatrix(x):
    return False
'@
        Write-Host "     scikit-learn 除去 + scipy をスタブ化" -ForegroundColor Green
    } else {
        if (Test-Path $sklearnDir) {
            Write-Host "     scipy は既にスタブ / sklearn を除去" -ForegroundColor Green
        } else {
            Write-Host "     scipy スタブ済み・sklearn 不在（変更なし）" -ForegroundColor DarkGray
        }
    }
}

$after = Get-DirSizeMB $PythonEmbed
Write-Host ("     pruning: {0} MB -> {1} MB (削減 {2} MB)" -f `
    $before, $after, [math]::Round($before - $after, 1)) -ForegroundColor Green
