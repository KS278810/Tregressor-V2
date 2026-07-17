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
| 入手方法 | [GitHub Releases](../../releases) から `T-regressor-*-win64.zip` をダウンロード、または `.\build_portable.ps1` でローカルビルド | `web/` 一式をホスティング、またはフォルダごとコピーして `offline.html` を開く |

### 入手方法（詳細）

`dist_portable/T-regressor/T-regressor.exe`・`dist_portable/T-regressor/native_dist/predict_native.exe`・
`web/offline-embed.js`（約59MB）は**いずれもビルド生成物のためgit管理下に置いていない**
（[`.gitignore`](.gitignore)参照。以前はexe 2本をコミットしていたが、コミット済みexeが
現行ソースと乖離する事故が起きたことと`.git`の肥大化を避けるため、2026-07にビルド生成物へ
統一した）。

- **exe版を使いたいだけの場合**: タグ(`v*`)push時に [`.github/workflows/windows-build.yml`](.github/workflows/windows-build.yml)
  が自動ビルドし、[GitHub Releases](../../releases) に配布zipをアップロードする。そこから
  `T-regressor-*-win64.zip` をダウンロードして展開すればよい。
- **自分でビルドしたい場合**: 下記「開発者向け：ビルド方法」の手順に従う。
- **Web版のオフライン配布(`offline.html`)一式を作りたい場合**: `web/offline-embed.js` を
  `cd web && node build_offline.mjs` で生成してから `web/` フォルダごとコピーする
  （`web/index.html`によるHTTP版のホスティングだけなら`offline-embed.js`は不要）。

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
  **exe版・Web版とも、選ばれたモデルがどの種類であっても**そのまま書き出せる
  （多項式Ridge・Blend(アンサンブル)含め全種別対応、2026-07-15〜。詳細は「動作の仕組み」節を参照）
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
  (GBDT/RandomForest/ExtraTrees)・ガウス過程・MLP・アンサンブル(Blend)の全モデル種別を
  `train_bridge.py`（ルート/`web/py/`共通）が書き出せる（2026-07拡張。詳細は
  [`web/js_predict_poc/README.md`](web/js_predict_poc/README.md)）。**読み取り側もexe版・
  Web版とも全6種別対応**（exe版C++予測エンジン`native_predictor/predict_native_v2.cpp`は
  2026-07-15に多項式Ridge/Blend(入れ子.tregの再帰読み込み)を追加し、Web版JS予測エンジンと
  対応範囲が揃った）

## 既知の制約

- **exe版とWeb版の実行環境（Pythonバージョン・数値ライブラリバージョン）は完全には一致しない**。
  exe版は `python-embed`（[`requirements-embed.txt`](requirements-embed.txt)で固定、Python 3.11.15 +
  numpy 2.3.5 / pandas 2.3.3 / lightgbm 4.6.0 等）、Web版はPyodideが同梱するPython/numpy
  （バージョンはPyodideのリリースに追従し、execと厳密には異なる）で動作する。学習ロジック本体
  （`train_bridge.py`/`_light.py`/`predict_template.py`）はexe版・Web版で完全に同一のコードだが、
  下回りのPython/numpyのバージョン差に起因する浮動小数点演算のごく僅かな丸め誤差までは
  排除できない。両者を実データで自動的にクロス検証するテストは現状ない
  （個別の予測エンジン3実装(C++/JS/Python)間のパリティは`web/js_predict_poc`のテストで
  担保しているが、これは「.tregを読む3実装同士」の閉じた比較であり、「exe版の学習結果」と
  「Web版の学習結果」を直接比較するものではない）。

## 書き出しEXEとウイルス対策ソフト

`学習済モデルのDL`（exe版）で書き出す予測専用EXE（`predict_native.exe`ベース、[`.treg`をEXE末尾に埋め込み](docs/treg-format.md)）は、
一部のウイルス対策ソフト（特にトレンドマイクロ系）で誤検知（False Positive）される場合があります。
**方針として、単体EXE配布の利便性（インストール不要・Pythonの知識がない相手にも渡せる）を優先し、
誤検知はAV側への報告や運用で対処します**（EXE自体の設計を変えて回避することはしません）。

### 検知される理由（正直な説明）

- **無署名**（コード署名証明書は未導入。[既知の制約](docs/treg-format.md#既知の制約-書き出しexeは常に無署名)参照）
- **静的リンクされたC++バイナリ**（`native_predictor/build_native.ps1`が`-static`でビルド。
  外部DLLへの依存が無い代わりに、一般的な小さいツール向けEXEより実行コードの塊が大きく、
  ヒューリスティック検知の「見慣れない構造」判定に引っかかりやすい）
- **末尾に任意バイト列（`.treg`モデルデータ）を追記する構造**（[EXEテール形式](docs/treg-format.md#exeテール形式exe版のみ)）。
  マルウェアがペイロードをファイル末尾に隠す手口と機械的なパターンが似ており、
  シグネチャ/ヒューリスティック双方で疑われやすい
- **低prevalence（世に流通している量が少ない）**。クラウドレピュテーション型の検知は
  「多くの人が使っている＝安全の傍証」というスコアリングをするため、ニッチなツールは
  実害の有無に関わらず疑わしさが増す

いずれも「実際に悪性コードが入っている」ことを意味しません。`native_predictor/predict_native_v2.cpp`
はソースを公開しているソースコードそのままビルドしたものであり、外部通信・自己増殖・
他プロセスへの注入等は一切行いません。

### 誤検知の確認手段（SHA256添付運用）

配布するEXEが改変されていないことを確認できるよう、リリースごとに公式ビルドのSHA256を
明記します（[GitHub Releases](../../releases)のリリースノート、または `T-regressor.exe`・
`predict_native.exe`と同じフォルダに`SHA256SUMS.txt`を添付する運用）。ハッシュ値は
以下のコマンドで確認できます:

```powershell
Get-FileHash .\T-regressor.exe -Algorithm SHA256
Get-FileHash .\native_dist\predict_native.exe -Algorithm SHA256
```

ウイルス対策ソフトが警告を出した場合は、まず上記ハッシュ値がリリースノート記載の値と一致するかを
確認してください。一致していれば公式ビルドそのものであり、警告はAV側の誤検知の可能性が高いです
（一致しなければ改変された/破損したファイルの可能性があるため、実行せず再ダウンロードしてください）。

### 誤検知の報告先

- **トレンドマイクロ**: [サンプル提出手順（Help Center）](https://helpcenter.trendmicro.com/en-us/article/tmka-14388)、
  または[サポートポータル](https://success.trendmicro.com/)から新規ケースを起票し、
  「Virus False Alarm（誤検知）」として該当EXEを添付する
- **Microsoft Defender / WDSI**: [ファイル誤検知申請ページ](https://www.microsoft.com/en-us/wdsi/filesubmission)
  から該当EXEをアップロードし、「Software developer」枠で誤検知（false positive）として申請する

いずれも解析には数日〜かかる場合があります。誤検知が確定した場合、各社側の定義データベースが
更新されるまでは個々の利用者側で該当ファイル/フォルダを除外設定する必要がある点はご了承ください。

### 代替手段: Web版のHTML書き出しはAV検知対象になりにくい

上記の誤検知を避けたい場合、**Web版（`web/index.html`または`web/offline.html`）の
`学習済モデルのDL`で書き出す単体HTML（`predict_template.html`に`.treg`をBase64埋め込み）は
実行ファイルではないため、通常のウイルス対策ソフトのスキャン対象になりません**
（Windowsの「安全でない実行ファイル」判定=MOTW/SmartScreenの対象外でもあります）。
Pythonの知識がない相手に予測専用ファイルを配りたいだけで、exe化そのものが目的でない場合は
Web版のHTML書き出しを検討してください（ブラウザで開くだけで動作します）。

## 開発者向け：ビルド方法

### exe版

前提環境:
- Rust（[rustup](https://rustup.rs/)）＋ `cargo install tauri-cli --version "^2"`
- MinGW-w64（`g++` / `windres`。ネイティブ予測EXEのビルドに使用）
- Python 3.11 embeddable package（[python.org](https://www.python.org/downloads/windows/) から取得）

```powershell
# 1. Python embeddable package (64-bit, Python 3.11) を配置し、依存パッケージを入れる
mkdir dist_portable\T-regressor\python-embed
# ここに embeddable package を展開したうえで:
#   - ダウンロードしたzipのSHA256をpython.orgのリリースページ記載値と必ず照合する
#   - get-pip.py (https://bootstrap.pypa.io/get-pip.py) で pip を導入
#   - python311._pth の `#import site` のコメントを外す（--target 展開先を認識させるため）
# 依存バージョンは requirements-embed.txt に固定してある（再現可能なビルドのため）:
.\dist_portable\T-regressor\python-embed\python.exe -m pip install -r requirements-embed.txt --target dist_portable\T-regressor\python-embed\Lib\site-packages

# 2. ネイティブ予測 EXE をビルド
.\native_predictor\build_native.ps1

# 3. アプリ本体をビルドし、配布フォルダ一式を組み立てる（[2/4]でpruning、[1/4]でimport検証も実施）
.\build_portable.ps1
```

成功すると `dist_portable\T-regressor\T-regressor.exe` に、Python環境ごと自己完結した配布用アプリが出来上がります。
Node.js / npm は exe版のビルドには不要です。詳細な自動セットアップ手順・依存の固定理由は
[`requirements-embed.txt`](requirements-embed.txt) と `build_portable.ps1` 冒頭のコメントを参照してください。
`build_portable.ps1` はPowerShell 7 (pwsh) での実行を前提としています（PowerShell 5.1では
ネイティブコマンド呼び出しの一部が誤動作することがあります）。

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

`train_bridge.py` 等がルートと `web/py/` に重複して存在するのは意図的です（exe版とWeb版で
Pythonランタイムの配置場所が異なるための複製）。2026-07の同期以降、**両ファイルは内容が
完全に一致**しており、多項式Ridge/Blendを含む全6モデル種別の `.treg` 書き出しをどちらも行います。
読み取り側（exe版の予測エンジン `native_predictor/predict_native_v2.cpp` とWeb版JS予測エンジン
`web/js_predict_poc/predict-core.js`）も2026-07-15より全6種別に対応しており、exe書き出し
（`export_robot`）・HTML書き出しのどちらでも、学習結果がどのモデル種別であってもそのまま
配布できます。学習ロジック本体（特徴量生成・CV・ハイパラ探索など）は両者で同一です。
両ファイルの差分は `diff train_bridge.py web/py/train_bridge.py` で随時確認できます。

## ライセンス

[MIT License](LICENSE)
