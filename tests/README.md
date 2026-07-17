# T-regressor 検証スイート

コード改修後の**回帰検出**用テスト。`dist_portable\T-regressor\python-embed\python.exe` で
実行するが、`verify_rebuild.py`/`test_harness.py` はいずれもプロジェクトルート直下の
`train_bridge.py`/`predict_template.py` を直接呼び出す（exeに埋め込まれたコピーではない）ため、
**`.py`ファイルだけの改修であれば `build_portable.ps1` でのリビルドは不要**で、そのまま実行できる。
リビルドが必要なのは、`native_predictor/`（C++、`predict_native.exe`）を変更した場合や、
`python-embed` 自体のセットアップ（依存パッケージ等）をやり直す場合のみ。

## 実行方法

プロジェクトルート（`15_TregV2`）から:

```powershell
$py = ".\dist_portable\T-regressor\python-embed\python.exe"
& $py tests\verify_rebuild.py      # 主回帰スイート（必須）
& $py tests\test_harness.py        # 詳細統合テスト
```

いずれも成功で終了コード `0`、失敗があれば `1` を返し、末尾に FAIL 一覧を出す。

## 各スイートの役割

### `verify_rebuild.py` — 主回帰スイート（改修後は必ず実行）
外部テストデータ（学習に使わない別seed）での **ground truth R²** で挙動を検証する。
「テストが通る」だけでなく「精度が実際に出ている／退行していない」まで見る。

- **quick不変**: easy300 で FE が走らない・`.treg` が v3・外部R²>0.95・時間<15s
- **thorough改善**: hard600 で外部R²が quick+0.02 以上（自動FE・ランダムサーチ・6モデルblendの効果）
- **native parity**: deployモデルと同型のとき Python予測と native exe予測が一致（rel<2e-3）。
  `predict_native.exe`はtype0-3(linear/lgbm/gp/mlp)のみ対応のため、デプロイモデルが
  type4/5(linear_poly/blend)になった場合はnative検証自体をスキップする(高-N2)
- **エッジ**: 15行thorough / 純ノイズ / 整数歪みy（log1p+smear+round+FE+native）

所要 約3〜5分（thorough学習を複数回走らせるため）。

### `test_harness.py` — 詳細統合テスト（25項目）
CLI境界・バイナリ整合・エラー処理を細かく突く。

- RESULT_JSON が構文的にパース可能であることの確認（json.loadsはNaN/Infinityをデフォルトで
  受理するため、NaN/Inf混入自体の検出はできない点に注意。それらの排除は
  train_bridge.py側のisfiniteガードが担う）、`trained_model_tmp` 残置なし
- **C1回帰**: LGBM-best データで native出力が定数化しないこと（`Tree=N` パーサの回帰テスト）
- **C2回帰**: 15特徴量スクリーニング時に `.treg` の n_feat がモデル実次元と一致
- **C3**: Blend の in-app 予測が学習時 OOF と整合（best が Blend のとき）
- エッジ6種: 15行 / target NaN / 文字列target / 不在target名 / 整数丸め / 12行(旧KFold全滅)

## 生成物

`tests/verify_work/` と `tests/_work_harness/` に一時CSV・モデル複製を作る（各実行で作り直す）。
コミット不要。

## `run_benchmark.py` — リリース前ベンチマークスイート（`tests/benchmarks/`）

`verify_rebuild.py`/`test_harness.py`が「コード改修直後の回帰検出」用なのに対し、こちらは
「リリース判断」用の網羅ベンチマーク。`tests/benchmarks/gen_datasets.py`が生成した13種の
合成データセット（線形/非線形/カテゴリ低・高カーディナリティ/bool混在/欠損重/歪みy/
外れ値/小データ/リーク検知番犬(pure_noise)/重複行/日本語列名(UTF-8・Shift-JIS)）を
quick・thorough両モードで学習し、`tests/benchmarks/baseline.json`との比較でR²退行・
3エンジン(C++/JS/Python)パリティ不一致等を検出する。

```powershell
& $py tests\run_benchmark.py --mode quick       # CI相当・数分
& $py tests\run_benchmark.py --mode thorough    # ローカル専用・十数分規模
```

詳細な合否基準・運用手順は [`docs/release-checklist.md`](../docs/release-checklist.md) を参照。
