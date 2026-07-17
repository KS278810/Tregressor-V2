# T-regressor 最終監査レポート（2026-07-17）

「最終版」確定前の徹底監査。4系統（未コミット変更セット専任／ML学習コード／予測エンジン・テスト体制／フロントエンド・Tauri）で並列実施し、テストは当環境で実走した。

## まず重要な事実: 作業ツリーに大型の未コミット変更がある

「最終版」と認識されている状態は、実は**未コミットの新機能2本**（13ファイル +2,729/-887、新規2ファイル）を含む:

- **i18n対応**: PROGRESS/ERRORをキー化しフロント辞書（ja/en）で翻訳。設定に言語切替。`data_warning`は互換のため不変
- **Web版Worker化＋async化**: HTTP配信版はPyodideをWeb Workerで実行（フリーズ解消）、offline(file://)版はWorker制限を回避してメインスレッド＋息継ぎ改善。学習ロジックの数値は不変

監査の結論、この変更セットは**中途半端な作業途中ではなく実質完成品**（設計一貫・辞書全キー完備・複製同期・生成物再生成・exeリビルドまで済み）。ただし下記の修正をしてからコミットすべき。

## 実走検証（全て当環境で確認）

npm test **50/50設定×302行=15,100件 全PASS**（HTML版・エンコーディング版も分割実行で全PASS）／破損.treg **803/803クラッシュなし**／`g++ -Wall -Wextra` **警告ゼロ**／root↔web/py **3ファイルバイト一致**／web生成物 **frontendから再生成済みを確認**／baseline.json 26件健全（**pure_noise番犬はR²≈0.00で機能**）／XSS: 新規+810行にinnerHTML流入なし。

## コミット前に必須（5件）

1. **treg-worker.js / treg-worker-client.js の git add 漏れ防止**。web/index.html:1410 がこの**未追跡2ファイル**を参照しており、add忘れでpushするとGitHub Pages配信のHTTP版が404で学習不能になる
2. **予測エラーが生キー表示になる退行**（frontend/index.html:2525）: `pebMsg.textContent = String(e)` が翻訳器を通らず、ユーザーに `PREDICT_ERROR:predict_failed:{"detail":...}` が生表示される。旧版は日本語文が読めたので明確な後退。`translateBackendMessage()` を通す1行修正＋web再生成（`parseKeyedMessage`に`Error: `プレフィックス除去も追加）
3. **linear_polyのOOF経路がfold内化から取り残されている**（train_bridge.py:972-994）: コメントが認める「top_feats相関選択」だけでなく、**target encoding済みdf_clippedを使うため自分のyが特徴に焼き込まれた状態でOOF評価**される（+FE派生列も同様）。中カーディナリティのカテゴリ列を持つ小規模データ（n<50はquickでも到達）でlinear_polyのR²だけ楽観化しモデル選択を歪める。`_fold_frame`を呼ぶ約20行の修正
4. **BOM付きUTF-8 CSVで先頭列がターゲット指定不能**（train_bridge.py:497-513）: pandasが先頭列名を`﻿col`で返し、`str.strip()`はU+FEFFを除去しない。**Excelの「CSV UTF-8」既定出力で発生**。`encoding='utf-8'`→`'utf-8-sig'`の1語修正
5. **CRLFチャーン2ファイルを戻す**（web/js_predict_poc/README.md・_manifest.json、改行のみの548行差分）＋コミットは `feat:`(i18n)/`feat:`(Worker)/`test:` に分割

## Medium（リリース後の早期パッチで可、優先順）

| # | 内容 | 場所 |
|---|------|------|
| M-1 | 数値文字列カテゴリのin-app vs 配布物乖離: Pythonはpandasパース後の値("1.5")、C++/JSはCSV生文字列("1.50")で照合するため、数値見かけのカテゴリでtarget mapミス。数値文字列カテゴリのフィクスチャも未整備 | predict_template.py:387 vs cpp:603/js:289 |
| M-2 | カテゴリsource列がCSVに無い場合の実装差: Pythonはmedian補完へ、C++/JSは`__NaN__`センチネル照合。3実装で仕様を統一すべき（センチネル側へ） | predict_template.py:384 |
| M-3 | bool列のclass_value "True" と Excel系 "TRUE" の不一致 → 配布物で全ゼロ(未知扱い)。one-hot生成時に正規化を | train_bridge.py:762-788 |
| M-4 | Workerクラッシュ後の自己修復が不完全（再initせず「エンジン未初期化」で失敗し続ける）。train/predict冒頭で`if(!_ready) await initEngine({})` | treg-worker-client.js:43-51 |
| M-5 | module worker非対応環境のフォールバックなし（メインスレッド直実行への縮退を追加） | treg-worker-client.js:11 |
| M-6 | D&D: dprが配線時1回取得でDPI混在モニタ間移動に追従しない（毎回読むだけ）／候補3がタイトルバー高さ未補正 | frontend/index.html:1296, 1311 |
| M-7 | windows-build.ymlがmain直pushで走らない（PR/タグ/手動のみ）。`push: branches:[main]`追加 | windows-build.yml:21-25 |
| M-8 | test_corrupt_treg.py がCI・release-checklist未組込（44秒で完走するのでpredict-parity.ymlに1ステップ） | — |
| M-9 | 数値文字列90%判定の分母に欠損が入り、欠損多めの数値列がカテゴリ誤判定→列除外に転落しうる。分母を非欠損に | train_bridge.py:767-769 |
| M-10 | 「ブラウザが固まる」系文言・コメントがWorker化後のHTTP版では虚偽（プラットフォーム能力で出し分け） | frontend/index.html:806-813ほか |

## Low（バックログ、抜粋）

ベンチ判定の一律0.02閾値はoutlier_contaminated(r2_std=0.14)でflakeの恐れ／run_benchmarkの1データセット失敗でスイート全体abort／baseline upsertの幽霊エントリ／e2e出力の非クォート／inf文字列フィクスチャ未整備／e2e許容誤差1e-4はGPがbestになると偽陽性の恐れ（2e-3へ）／ドキュメントの設定数陳腐化（実体50、READMEは46、CLAUDE.mdは40）／`_candidate_r2_std`計算不能時0.0→Noneへ／ES専用valの最小行数ガード／学習中ドロップが無言無視／lib.rs由来クラッシュ文言の非キー化／未知PROGRESSキーの生表示（キー追加時の辞書同時更新をCLAUDE.mdルールに）／modelZipBytesの無駄転送／巨大CSV×fold内化のメモリ増（fold数分のDataFrameコピー、行数上限ガードなし）／offline-embed.js 59MBのgit管理（LFS/CI生成化）。

## さらなる改善レバー（優先順）

1. **inner-OOF target encoding**: fold-train内TE値が自身のyで計算されており、木モデルがTE列に過剰依存。fold-train内をさらにK分割してout-of-fold TEにするのが定石。効果中〜大
2. **Web版の学習キャンセル**: Worker化により`worker.terminate()`+再生成で実装可能になった（現状Web版はCANCEL非表示）。M-4とセットで
3. **デプロイ用前処理統計の100%再fit**: 現在fold0-train(90%)でfitした統計を最終モデルに使用。デプロイ用のみ全行で再fitが定石
4. **見送り6件の優先順位**: MLP fold内ES（.treg変更不要・効果/コスト最良）→ quick OOF化（n<200まで拡大の折衷案）→ LGBM NaN直渡し（効果最大だが.treg v6+4点セットで最重量、次期メジャー）→ X歪み変換 → Blend切片 → GP Matern
5. repeated CV（n<100で2リピート）、Worker初期化の先行ウォームアップ、web/index.htmlへのCSP meta（Pages多層防御）

## GitHub Pages配信の注意（8abfb41関連）

`.nojekyll`の目的（`_light.py`がJekyllに除外されるのを防ぐ）は正しいが、**Pagesの公開ソースが`web/`をサイトルートとして配信する設定であることを要確認**（リポジトリルート配信なら`.nojekyll`の位置が誤り）。パスは全て相対でサブパス配信OK、Pyodide完全同梱でCDN非依存、外部通信ゼロは維持されている。

## 総合判定

**Critical級の数値バグ・データ破壊・セキュリティ問題は無し。** 予測パリティ・.tregフォーマット・ベンチ体制はリリース品質。ただし「必須5件」（うち3件はコードで、修正量は合計30行程度＋再生成）を処理してからコミット・確定すること。特に#1（worker 2ファイルのadd漏れ）は事故りやすいので注意。

### 確定までの手順

1. 必須5件を修正（下のPhase 10プロンプト）
2. コミット（feat: i18n / feat: Worker / fix: 修正群 / test: に分割）
3. npm test・test_light_parity・verify_rebuild・test_harness・run_benchmark（quick）を再実行
4. 実機確認: 言語切替、Web版(HTTP)の学習がフリーズしないこと、offline.htmlの学習、予測エラー時の表示が翻訳されること
5. 確定タグ → バックアップ → git filter-repo → 署名調達（既定の人間タスク）

## Phase 10 修正プロンプト（CC用）

```
T-regressorの最終監査(FINAL_AUDIT_2026-07-17.md)で見つかった必須5件を修正します。CLAUDE.mdを読んでから着手してください。現在の作業ツリーには未コミットのi18n/Worker化変更があり、これを壊さずに修正を加えます。

1. web/treg-worker.js と web/treg-worker-client.js を git add する（web/index.htmlが参照する未追跡ファイル。add漏れするとHTTP配信版が404で壊れる）
2. frontend/index.html:2525付近のonFail: `pebMsg.textContent = String(e)` を `translateBackendMessage()` 経由に修正。parseKeyedMessageの入口で `/^Error:\s*/` を除去（Web版のError文字列化対策）。修正後 web再生成
3. train_bridge.py:972-994 の linear_poly OOF分岐を _fold_frame ベースに修正: 非poly分岐と同様に fold毎の df_all_per_fold を使い、top_feats も fold-train のみで _poly_top_feats_by_corr を再選定する（最終モデル用top_featsは現行のままdf_full系でよい。ただしdf_trainでfitしたencoder/recipe適用のものを使う）。修正後、web/py/へ同期し、tests/run_benchmark.py --datasets categorical_high,small_n --mode both で数値変化を確認、変化があればbaseline更新（コミットメッセージに前後値を記録。リーク除去による低下は正当）
4. train_bridge.py の _read_csv_with_encoding_fallback: utf-8 を utf-8-sig に変更（BOM付きExcel CSV UTF-8対応）。BOM付きCSVの読み込みテストを tests/ に追加。web/py/同期
5. web/js_predict_poc/README.md と matrix/_manifest.json のCRLFのみの差分を `git checkout --` で破棄

完了後: npm test 全PASS、root↔web/py diff ゼロ、web生成物再生成を確認し、コミットを feat:(i18n) / feat:(Worker+async) / fix:(監査修正群) / test: に分割して作成。既存の未コミット変更のコミット分割も同時に行うこと。verify_rebuild.py / test_harness.py はWindows側で実行して結果を報告。
```
