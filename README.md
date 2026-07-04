# T-regressor

**CSVをドラッグ&ドロップするだけで回帰モデルを自動学習・予測できる、Windows用デスクトップアプリ。**

プログラミング知識がなくても、表形式データ（CSV）から数値を予測するAIモデルを作れることを目指したツールです。学習・予測・単体実行ファイルとしての配布まで、すべてGUIの操作だけで完結します。

## 特徴

- **CSVドロップだけで学習** — 予測したい列を選んでボタンを押すだけ
- **4種類のモデルを自動比較** — 線形回帰（Ridge）・LightGBM・ガウス過程（GP）・ニューラルネット（MLP）を学習し、最も精度の良いものを自動選択
- **2つの学習モード**
  - **お急ぎ** — 4モデルを並列実行し、数秒〜十数秒で結果を得る
  - **じっくり** — 交差検証・自動特徴量生成・ハイパラ探索・複数モデルのアンサンブル（Blend）で精度を追求
- **学習済みモデルを単体EXE化** — Pythonの知識がない相手にも、ダブルクリック（またはCSVをドラッグ&ドロップ）するだけで使える予測専用の実行ファイルを書き出せる
- **Python不要で配布可能** — 実行環境（Python）を内包した自己完結型アプリ。配布先にPythonをインストールする必要はない
- **高速なネイティブ推論** — 予測専用の実行ファイルはC++で実装されており、Pythonを介さず高速に動作する

## 使い方

1. `T-regressor.exe` を起動する
2. 学習に使うCSVファイルを ①の枠にドラッグ&ドロップする
3. 予測したい列（ターゲット列）を選び、`TRAIN` を押す
4. 学習が終わったら、③の枠に新しいCSVをドロップすると予測結果が得られる
5. `学習済モデルのDL` から、予測専用の実行ファイルを書き出せる（配布用）

## 動作の仕組み（技術的な概要）

- **フロントエンド**: HTML/CSS/JavaScript（フレームワーク不使用、ビルド工程なし）
- **アプリ基盤**: [Tauri v2](https://v2.tauri.app/)（Rust）
- **学習処理**: Python（LightGBM・numpy・pandasのみに依存。scikit-learn・scipy相当の処理は依存削減のため自前実装（[`_light.py`](_light.py)）に置き換え済み）
- **予測処理（配布用EXE）**: 外部依存ゼロのC++。学習済みモデルは独自バイナリ形式（`.treg`）でEXEの末尾に埋め込まれる
- **配布形態**: Python実行環境ごと自己完結させた1つのEXE。初回起動時に内部展開される

## 開発者向け：ビルド方法

### 前提環境

- Rust（[rustup](https://rustup.rs/)）＋ `cargo install tauri-cli --version "^2"`
- MinGW-w64（`g++` / `windres`。ネイティブ予測EXEのビルドに使用）
- Python 3.11 embeddable package（[python.org](https://www.python.org/downloads/windows/) から取得）

Node.js / npm は不要です（フロントエンドは素のHTMLで、ビルド工程を持ちません）。

### 手順

```powershell
# 1. Python embeddable package を配置し、依存パッケージを入れる
#    (scipy は lightgbm の依存として自動で入るが、後述の pruning で軽量スタブに置換される)
mkdir dist_portable\T-regressor\python-embed
# ここに embeddable package を展開したうえで:
.\dist_portable\T-regressor\python-embed\python.exe -m pip install numpy pandas lightgbm --target dist_portable\T-regressor\python-embed\Lib\site-packages

# 2. ネイティブ予測 EXE をビルド
.\native_predictor\build_native.ps1

# 3. アプリ本体をビルドし、配布フォルダ一式を組み立てる
.\build_portable.ps1
```

成功すると `dist_portable\T-regressor\T-regressor.exe` に、Python環境ごと自己完結した配布用アプリが出来上がります。

### テスト

改修後は `tests/README.md` の手順に従い、回帰テスト（`verify_rebuild.py`）と統合テスト（`test_harness.py`）を実行してください。精度の退行やnativeとPythonの予測乖離を検出します。

## プロジェクト構成

```
15_TregV2/
├── frontend/               フロントエンド (index.html + 画像/GIF素材)
├── native_predictor/       予測専用ネイティブEXE (C++)
├── src-tauri/              Tauri本体 (Rust)
├── tests/                  回帰・統合テスト
├── _light.py               numpyのみで実装した学習用ユーティリティ (sklearn/scipy代替)
├── train_bridge.py         学習処理のメインスクリプト
├── predict_template.py     予測処理のスクリプト
├── build_portable.ps1      配布フォルダ一式を組み立てるビルドスクリプト
└── prune_embed.ps1         Python実行環境の依存スリム化スクリプト
```

## ライセンス

[MIT License](LICENSE)
