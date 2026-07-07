# T-regressor

**CSVをドラッグ&ドロップするだけで回帰モデルを自動学習・予測できるツール。**

プログラミング知識がなくても、表形式データ（CSV）から数値を予測するAIモデルを作れることを
目指しています。学習・予測・配布用ファイルの書き出しまで、すべてGUIの操作だけで完結します。

**2つの配布形態**があり、どちらも見た目・操作感・学習/予測ロジックは共通です
（[`frontend/index.html`](frontend/index.html) が唯一のフロントエンドソース）。

| | **exe版**（Windows デスクトップアプリ） | **Web版**（ブラウザで動作） |
|---|---|---|
| 実体 | [`dist_portable/T-regressor/T-regressor.exe`](dist_portable/T-regressor/T-regressor.exe) | [`web/index.html`](web/index.html)（要ホスティング）/ [`web/offline.html`](web/offline.html)（ダブルクリックのみ） |
| 実行環境 | Windows専用。Python環境ごと自己完結した単体EXE | ブラウザだけ（インストール不要）。学習・予測は [Pyodide](https://pyodide.org/)(WebAssembly) で端末内実行 |
| 外部通信 | なし | なし（ライブラリ一式を同梱、外部CDN不使用） |
| 学習・予測速度 | 速い（ネイティブPython/C++） | やや遅い（WebAssembly実行のオーバーヘッド） |
| 学習済みモデルの配布形式 | 単体 `.exe`（`predict_native.exe` に `.treg` を埋め込み） | 単体 `.html`（`predict_template.html` に `.treg` をBase64埋め込み） |
| 入手方法 | このリポジトリから `T-regressor.exe` をダウンロード | `web/` 一式をホスティング、またはフォルダごとコピーして `offline.html` を開く |

## 特徴

- **CSVドロップだけで学習** — 予測したい列を選んでボタンを押すだけ
- **候補モデルを自動比較** — 線形回帰(Ridge)・多項式Ridge・LightGBM(GBDT/RandomForest/ExtraTrees)・
  ガウス過程(GP)・ニューラルネット(MLP)・それらのアンサンブル(Blend)を学習し、
  最も精度の良いものを自動選択
- **2つの学習モード**
  - **お急ぎ** — 主要モデルを並列実行し、数秒〜十数秒で結果を得る
  - **じっくり** — 交差検証・自動特徴量生成・ハイパラ探索・複数モデルのアンサンブル(Blend)で精度を追求
- **学習済みモデルを単体ファイル化** — Pythonの知識がない相手にも、ダブルクリック
  （またはCSVをドラッグ&ドロップ）するだけで使える予測専用ファイルを書き出せる。
  **選ばれたモデルがどの種類であっても**そのまま書き出せる（2026-07〜）
- **インストール不要で配布可能** — exe版はPython環境ごと自己完結、Web版はブラウザのみで動作
- **サーバー送信なし** — 学習データ・モデルはどちらの版でも利用者の端末外に出ない

## 使い方

### exe版

1. `T-regressor.exe` を起動する
2. 学習に使うCSVファイルを ①の枠にドラッグ&ドロップする
3. 予測したい列（ターゲット列）を選び、`TRAIN` を押す
4. 学習が終わったら、③の枠に新しいCSVをドロップすると予測結果が得られる
5. `学習済モデルのDL` から、予測専用の実行ファイル(`.exe`)を書き出せる（配布用）

### Web版

1. `web/index.html` をホスティングして開く（ローカル確認は `cd web && node serve.mjs`）、
   または `web/offline.html` をフォルダごとコピーして直接ダブルクリックする
2. 操作感はexe版と同じ（CSVドロップ → TRAIN → 予測 → `学習済モデルのDL`）
3. `学習済モデルのDL` では、ブラウザだけで開ける単体HTML(`predict_template.html`に`.treg`埋め込み)を書き出す

Web版の詳細（2つの配布形態の違い、公開手順、既知の制約など）は [`web/README.md`](web/README.md) を参照。

## 動作の仕組み（技術的な概要）

- **フロントエンド**: HTML/CSS/JavaScript（フレームワーク不使用、ビルド工程なし）。
  exe版・Web版で共通のソース（[`frontend/index.html`](frontend/index.html)）から、
  `IS_TAURI` 分岐の `Platform` 抽象化層でバックエンド呼び出し
  （Tauri invoke/event ↔ Pyodide直接呼び出し）だけを切り替える
- **アプリ基盤(exe版)**: [Tauri v2](https://v2.tauri.app/)（Rust）
- **アプリ基盤(Web版)**: [Pyodide](https://pyodide.org/)（WebAssembly上のCPython）。詳細は [`web/README.md`](web/README.md)
- **学習処理**: Python（LightGBM・numpy・pandasのみに依存。scikit-learn・scipy相当の処理は
  依存削減のため自前実装（[`_light.py`](_light.py)）に置き換え済み）。exe版・Web版で完全に同一のコード
- **予測処理（配布用ファイル）**:
  - exe版: 外部依存ゼロのC++（[`native_predictor/`](native_predictor/)）。学習済みモデルは
    独自バイナリ形式（`.treg`）でEXEの末尾に埋め込まれる
  - Web版: 依存ゼロのJavaScript（[`web/js_predict_poc/predict-core.js`](web/js_predict_poc/predict-core.js)
    と数値的に同一のロジックが `predict_template.html` にインライン化）。`.treg` はBase64化してHTML内に埋め込む
- **`.treg`形式**: 独自のコンパクトバイナリ形式。線形(Ridge)・多項式Ridge・LightGBM系
  (GBDT/RandomForest/ExtraTrees)・ガウス過程・MLP・アンサンブル(Blend)の全モデル種別に対応
  （2026-07拡張。詳細は [`web/js_predict_poc/README.md`](web/js_predict_poc/README.md)）

## 開発者向け：ビルド方法

### exe版

前提環境:
- Rust（[rustup](https://rustup.rs/)）＋ `cargo install tauri-cli --version "^2"`
- MinGW-w64（`g++` / `windres`。ネイティブ予測EXEのビルドに使用）
- Python 3.11 embeddable package（[python.org](https://www.python.org/downloads/windows/) から取得）

```powershell
# 1. Python embeddable package を配置し、依存パッケージを入れる
mkdir dist_portable\T-regressor\python-embed
# ここに embeddable package を展開したうえで:
.\dist_portable\T-regressor\python-embed\python.exe -m pip install numpy pandas lightgbm --target dist_portable\T-regressor\python-embed\Lib\site-packages

# 2. ネイティブ予測 EXE をビルド
.\native_predictor\build_native.ps1

# 3. アプリ本体をビルドし、配布フォルダ一式を組み立てる
.\build_portable.ps1
```

成功すると `dist_portable\T-regressor\T-regressor.exe` に、Python環境ごと自己完結した配布用アプリが出来上がります。
Node.js / npm は exe版のビルドには不要です。

### Web版

前提環境: Node.js のみ（フロントエンドはビルド工程なしの素のHTML/JS）。

```bash
# frontend/index.html を編集したら、Web版を最新化するために必ず実行
cd web
node build_frontend.mjs

# Pyodide本体・Pythonソース・predict_template.html を変更したら、
# ダブルクリック版の埋め込みデータも再生成
node build_offline.mjs
```

詳細は [`web/README.md`](web/README.md) を参照。

### テスト

- **exe版**: 改修後は `tests/README.md` の手順に従い、回帰テスト（`verify_rebuild.py`）と
  統合テスト（`test_harness.py`）を実行してください。精度の退行やnativeとPythonの予測乖離を検出します。
- **Web版**: `cd web/js_predict_poc && npm install && npm test` で、JS予測エンジンとC++参照実装の
  数値一致（40設定×302行）を検証します。push/PR時にGitHub Actions（`.github/workflows/predict-parity.yml`）
  でも自動実行されます。

## プロジェクト構成

```
15_TregV2/
├── frontend/               ★共通フロントエンドソース (index.html + 画像/GIF素材)
│                           exe版・Web版はここから生成される。直接編集するのはここだけ
├── native_predictor/       予測専用ネイティブEXE (C++)。exe版の配布用バイナリ兼、
│                           Web版JS予測エンジンの数値検証リファレンス
├── src-tauri/              Tauri本体 (Rust)。exe版のアプリ基盤
├── dist_portable/          exe版のビルド出力 (T-regressor.exe を含む)
├── web/                    Web版一式 (index.html / offline.html / Pyodide同梱 / JS予測エンジン)
│                           詳細は web/README.md 参照
├── tests/                  exe版の回帰・統合テスト
├── _light.py               numpyのみで実装した学習用ユーティリティ (sklearn/scipy代替)
├── train_bridge.py         学習処理のメインスクリプト (exe版用。Web版は web/py/train_bridge.py)
├── predict_template.py     予測処理のスクリプト (exe版用。Web版は web/py/predict_template.py)
├── build_portable.ps1      exe版の配布フォルダ一式を組み立てるビルドスクリプト
└── prune_embed.ps1         Python実行環境の依存スリム化スクリプト
```

`train_bridge.py` 等がルートと `web/py/` に重複して存在するのは意図的です。exe版の予測エンジン
（`native_predictor/predict_native_v2.cpp`）は現状4モデル種別（linear/lgbm/gp/mlp）のみ対応のため、
ルート側は多項式Ridge/Blend等の新しい `.treg` 書き出しを行わない古い挙動のまま据え置いています
（`web/py/train_bridge.py` の方が新しい機能を持ちますが、それはWeb版のJS予測エンジンが全6種別に
対応しているためです）。学習ロジック本体（特徴量生成・CV・ハイパラ探索など）は両者で同一です。

## ライセンス

[MIT License](LICENSE)
