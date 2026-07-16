# T-regressor (TregV2)

CSVドロップだけで回帰モデルを自動学習・予測するツール。**exe版**（Tauri+Python embeddable、Windows専用）と
**Web版**（Pyodide/WebAssembly、ブラウザのみ）の2配布形態があり、見た目・操作感・学習/予測ロジックは共通。

詳細は [README.md](README.md) / [web/README.md](web/README.md) / [tests/README.md](tests/README.md) を参照。

## 運用ルール（必須）

### 1. フロントエンドは frontend/index.html のみ編集する

[`frontend/index.html`](frontend/index.html) が唯一のソース。exe版・Web版とも`IS_TAURI`分岐の`Platform`
抽象化層でここから生成される。編集後は必ず実行してWeb版を再生成すること:

```bash
cd web
node build_frontend.mjs
node build_offline.mjs
```

`web/index.html`・`web/offline.html`・`web/assets/`を直接編集しない（自動生成物、上書きされる）。

### 2. Pythonスクリプトはルートとweb/py/に複製がある

`train_bridge.py` / `_light.py` / `predict_template.py` はプロジェクトルートと `web/py/` に同一内容で
重複配置されている（exe版とWeb版でPythonランタイムの配置場所が異なるため）。**ルート側を変更したら
必ず同名ファイルを `web/py/` にもコピーし**、差分ゼロを確認する:

```bash
git diff --no-index train_bridge.py web/py/train_bridge.py
git diff --no-index _light.py web/py/_light.py
git diff --no-index predict_template.py web/py/predict_template.py
```

### 3. .tregバイナリ形式の変更は4点セット

`.treg`形式（linear/linear_poly/lgbm/rf/xt/gp/mlp/blendの全モデル種別対応）を変更する場合、以下を
**同時に**変更すること。片方だけの改修は3実装間の予測乖離を生む:

- `train_bridge.py`（writer。ルート/`web/py/`両方、ルール2参照）
- `native_predictor/predict_native_v2.cpp`（exe版予測エンジン）
- `web/js_predict_poc/predict-core.js`（Web版JS予測エンジン、PoC本体）
- `predict_template.py` の対応する `_predict_*` 関数、および `web/predict_template.html` の
  インライン化されたJS版（`predict-core.js`と同一ロジックを埋め込み）

加えて `web/js_predict_poc/matrix/` のパリティフィクスチャ（`.treg`サンプル・期待値CSV）を更新する。

### 4. テストコマンド

| 対象 | コマンド | 備考 |
|---|---|---|
| JS/C++/Python予測パリティ | `cd web/js_predict_poc && npm test` | 40設定×302行+エンコーディング境界。push/PR時にCIでも自動実行 |
| `_light.py` vs sklearn相当 | `python tests/test_light_parity.py` | sklearn/scipy入りのCPythonが必要（embeddable pythonでは不可） |
| exe版回帰テスト | `& $py tests\verify_rebuild.py` | `$py = dist_portable\T-regressor\python-embed\python.exe`。主回帰スイート、必須 |
| exe版統合テスト | `& $py tests\test_harness.py` | 25項目の詳細統合テスト |

`.py`だけの改修なら`build_portable.ps1`のリビルドは不要（exe埋め込みコピーでなくルート直下を直接呼ぶ）。
リビルドが必要なのは`native_predictor/`（C++）変更時、または`python-embed`セットアップ自体のやり直し時のみ。

### 5. コミットは変更種別ごとに分ける

`fix:` / `feat:` / `test:` / `build:` / `docs:` プレフィックスで分離する。フロントエンド編集+ビルド生成物+
Python複製同期のような一連の変更でも、性質が異なれば別コミットにする。

### 6. dist_portable/ は生成物、直接編集しない

`build_portable.ps1`が組み立てる配布フォルダ一式。ソースはリポジトリの他の場所にある。

### 7. 予測エンジン3実装の数値挙動を変える変更は3実装同時作業＋パリティ確認必須

予測ロジックは「ネイティブC++」「Python（Pyodide）」「JS」の3系統が並行して存在する。どれか1つを
変更したら、ルール3の4点セットを揃えたうえで必ず `cd web/js_predict_poc && npm test` を通し、
3系統の数値が一致していることを確認してからマージ・配布すること。
C++が非対応の型（現状 linear_poly/blend の一部検証）はPython独立実装との突合せで代替する
（`web/js_predict_poc/README.md`参照）。
