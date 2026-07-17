# リリース前チェックリスト

タグ(`v*`)push前に、開発者のローカル環境（Windows、`dist_portable\T-regressor\python-embed\`
が構築済み、`native_predictor\predict_native.exe`がビルド済み）で実行する手順と合否基準。
CIは一部（quickベンチマークのみ等）を自動実行するが、**thoroughモードのベンチマークと
`tests/verify_rebuild.py`/`test_harness.py`のフルスイートはCI未実施のためローカル実行が必須**。

## 実行コマンド一覧

プロジェクトルート（`15_TregV2`）から、PowerShellで:

```powershell
$py = ".\dist_portable\T-regressor\python-embed\python.exe"

# 1. JS/C++/Python予測パリティ(40設定×302行+エンコーディング境界)
cd web\js_predict_poc
npm test
cd ..\..

# 2. _light.py vs sklearn相当(システムPython。sklearn/scipy入りのCPythonが必要)
python tests\test_light_parity.py

# 3. exe版回帰テスト(主回帰スイート、必須)
& $py tests\verify_rebuild.py

# 4. exe版統合テスト(25項目)
& $py tests\test_harness.py

# 5. リリース前ベンチマークスイート(quick、CIと同等・数分)
& $py tests\run_benchmark.py --mode quick

# 6. リリース前ベンチマークスイート(thorough、ローカル専用・12データセットで十数分規模)
& $py tests\run_benchmark.py --mode thorough

# 7. Rust側コンパイルチェック(python-embed同梱なしの軽量確認)
cd src-tauri
$env:TREG_ALLOW_EMPTY_EMBED = "1"
cargo build
cd ..
```

コマンド5・6を初めて実行する環境、またはデータセット構成
（`tests/benchmarks/gen_datasets.py`）を変更した直後は、先に基準値を作る:

```powershell
& $py tests\benchmarks\gen_datasets.py   # tests/benchmarks/data/*.csv + manifest.jsonを生成
& $py tests\run_benchmark.py --update-baseline --mode quick
& $py tests\run_benchmark.py --update-baseline --mode thorough
```

`--update-baseline`は`tests/benchmarks/baseline.json`をデータセット単位でupsert更新するため、
`--datasets`で絞り込んで複数回に分けて実行しても既存の基準値を消さない
（thoroughは1データセットあたり数十秒〜2分程度かかるため、環境によっては
`--datasets name1,name2,...`で数回に分けて実行するとよい）。

## 合否基準

| チェック | 合格基準 | 不合格時の扱い |
|---|---|---|
| JS/C++/Python予測パリティ (`npm test`) | 全設定PASS | リリース禁止。3実装のいずれかが数値的に食い違っている |
| `_light.py` vs sklearn相当 | 全PASS | リリース禁止。自前実装がsklearn相当から数値的にズレている |
| `verify_rebuild.py` | 全PASS（quick不変・thorough改善・native parity・エッジケース） | リリース禁止 |
| `test_harness.py` | 全25項目PASS | リリース禁止 |
| `run_benchmark.py`(quick/thorough) | FAIL 0件（WARNは許容） | **FAIL** = リリース禁止。**WARN** = リリース可、ただし内容を確認しリリースノートに記載を検討 |
| `cargo build` | コンパイル成功 | リリース禁止 |

`run_benchmark.py`内部の個別基準（[`tests/run_benchmark.py`](../tests/run_benchmark.py)冒頭のdocstring参照）:

- baseline比でR²が**0.02超低下** → FAIL（精度退行の検出）
- `pure_noise`データセットのR²が**0.15超** → FAIL（学習パイプラインへのリーク混入の検知番犬。
  XとYが無関係な乱数であるにもかかわらず高いR²が出る場合、fold外fitのtarget encoding等の
  リークを疑う）
- best_modelの種別（`model_type`）がbaselineから変化 → WARN（精度が同等以上でモデル選択が
  変わることは正常にあり得るため、FAILにはしないが変化自体は把握しておく）
- 3エンジン（C++/JS/Python）の予測パリティ不一致 → FAIL
- 各データセット固有の期待特性（カテゴリ列検出、外れ値/重複行警告の発火など） → FAIL
- 所要時間（`elapsed_sec`）はJSON記録するがINFO表示のみで合否には使わない（負荷変動でflakeするため）

## WARN(best_model種別変化)が出た場合の判断

WARNは「壊れた」ではなく「選ばれるモデルが変わった」という情報。以下を確認してから
リリース判断する:

1. 該当データセット・モードのR²がbaselineから大きく変化していないか（`run_benchmark.py`の
   出力表で確認。R²自体は0.02以内の変動ならFAILにならないが、目視でも妙な値でないか見る）
2. 直近の変更内容が、その変化を説明できる意図的なものか（例: モデル選択ロジックの改修、
   ハイパラ探索範囲の変更など）
3. 説明できない変化であれば、`git bisect`等で原因コミットを特定してから判断する

## baseline.jsonを更新すべきタイミング

- 意図した精度改善・モデル選択ロジック変更を確認済みで、新しい数値を今後の基準値としたい時
- `tests/benchmarks/gen_datasets.py`のデータセット構成を変更した時（既存データセットの
  生成ロジックを変えた場合は当該データセットのみ、新規データセットを追加した場合は
  新規分のみ`--datasets`で絞り込んで追加すればよい）

`--update-baseline`実行時も学習失敗・パリティ不一致・データセット固有チェックの異常は
検出される（`FAIL`として表示される。ただし基準値のupsert自体は行われるため、`FAIL`が出た
状態のbaseline.jsonを誤ってコミットしないよう出力を確認すること）。
