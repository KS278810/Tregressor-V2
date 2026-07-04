# T-regressor 検証スイート

コード改修後の**回帰検出**用テスト。埋め込みPython（`dist_portable`）で実行するため、
改修したら **先に `build_portable.ps1` でリビルド** してから回す（スクリプトはexeに埋め込まれるため）。

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
- **native parity**: deployモデルと同型のとき Python予測と native exe予測が一致（rel<2e-3）
- **エッジ**: 15行thorough / 純ノイズ / 整数歪みy（log1p+smear+round+FE+native）

所要 約3〜5分（thorough学習を複数回走らせるため）。

### `test_harness.py` — 詳細統合テスト（25項目）
CLI境界・バイナリ整合・エラー処理を細かく突く。

- RESULT_JSON の strict parse（NaN/Inf混入検出）、`trained_model_tmp` 残置なし
- **C1回帰**: LGBM-best データで native出力が定数化しないこと（`Tree=N` パーサの回帰テスト）
- **C2回帰**: 15特徴量スクリーニング時に `.treg` の n_feat がモデル実次元と一致
- **C3**: Blend の in-app 予測が学習時 OOF と整合（best が Blend のとき）
- エッジ6種: 15行 / target NaN / 文字列target / 不在target名 / 整数丸め / 12行(旧KFold全滅)

## 生成物

`tests/verify_work/` と `tests/_work_harness/` に一時CSV・モデル複製を作る（各実行で作り直す）。
コミット不要。
