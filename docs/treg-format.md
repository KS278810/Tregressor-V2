# .treg バイナリフォーマット仕様

T-regressor の学習済みモデルを配布用に書き出す独自コンパクトバイナリ形式。書き出しは
`train_bridge.py`(ルート/`web/py/`に同一内容で複製、CLAUDE.mdルール2)、読み込みは以下の
3実装が並行して存在する:

- `native_predictor/predict_native_v2.cpp`(exe版、C++、外部依存ゼロ)
- `web/js_predict_poc/predict-core.js`(Web版、JS。`web/predict_template.html`にも同一ロジックが
  インライン化されている)
- `predict_template.py`(アプリ内Python予測。ルート/`web/py/`に複製)

この4点セットのいずれかを変更する場合は必ず同時に変更し、`web/js_predict_poc/matrix/`の
パリティフィクスチャを更新した上で`cd web/js_predict_poc && npm test`を通すこと
(CLAUDE.mdルール3・7)。本ドキュメントは `predict_native_v2.cpp` の読み込み実装と
`train_bridge.py` の書き出し実装(`_write_treg_stream`)を突き合わせて起草した。

## 基本規則

- **エンディアン**: すべて **リトルエンディアン固定**。C++側は`std::memcpy`によるネイティブ
  バイト列読み書きのため、ビッグエンディアン環境(実質存在しない)では動作しない。
  JS側は`DataView`の`little-endian`引数を明示指定。
- **浮動小数点**: モデル係数・スケーラ統計量など大半のペイロードは **`float32`**(4バイト)。
  median(中央値補完用)のみ**`float64`**(8バイト)で保存する(値の桁が大きくなりやすい生スケール
  データを保持するため)。yeo_johnsonのlambda・smear・y_clip・x_clip境界・y_transform関連の
  スカラー値も`float32`。
- **文字列**: `u16`(2バイト、リトルエンディアン)の長さプレフィックス + UTF-8バイト列
  (NUL終端なし)。65535バイトを超える文字列は書き出せない(列名・target名がこの制約を超えることは
  実務上ない)。
- **整数**: 符号なしは`u32`(4バイト)が基本(ノード数・特徴数・配列長など)。LGBMの
  `left_child`/`right_child`、poly項の`term_a`/`term_b`のみ符号付き`i32`(負値で「葉」「単項」を
  表現するため)。
- 可変長配列はすべて「`u32`件数 → 要素を件数分」の形。

## 全体レイアウト

```
[ヘッダ]
[v4+: 派生特徴(自動FE)ブロック]
[モデル種別ごとのペイロード(type 0〜5)]
[Y逆変換ブロック(v2+)]
[予測後処理ブロック(v3+)]
[共通テール: target_col / feat_cols / medians]
```

blend(type 5)のみ、上記のうち「モデル種別ごとのペイロード」部分に他モデルの完全な`.treg`
バイト列(ヘッダから共通テールまで丸ごと)を入れ子で埋め込む(後述)。

### ヘッダ(全バージョン共通)

| オフセット | 型 | 内容 |
|---|---|---|
| +0 | `char[4]` | マジック `"TREG"` (0x54 0x52 0x45 0x47) |
| +4 | `u8` | `file_version`(現行の書き出しは3または4。読み込み側は1〜4を受理し、5以上は明示エラーで拒否) |
| +5 | `u8` | `model_type`(下記表) |
| +6 | `u32` | `n_feat`(このモデルが直接使う特徴量数。**blendのみ0が正当値**— 各メンバーが個別に特徴を持つため) |

`model_type`:

| 値 | 種別 | 備考 |
|---|---|---|
| 0 | `linear` | Ridge回帰 |
| 1 | `lgbm` | LightGBM(GBDT本体、および`rf`/`xt`をエイリアスして書き出したもの。leaf値は`average_output`検出時に木本数で事前割り済み) |
| 2 | `gp` | ガウス過程(ARD-RBFカーネル) |
| 3 | `mlp` | 多層パーセプトロン(numpy自前実装) |
| 4 | `linear_poly` | 多項式Ridge(poly-Ridge。行数≤200かつ特徴数≤8の小データでのみ選ばれる) |
| 5 | `blend` | アンサンブル(他モデルの入れ子) |

`n_feat`検査: 非blend型は共通テールの`feat_cols`実要素数と一致することを読み込み側が強制する
(不一致は破損ファイルとみなしロード拒否。旧仕様の既知バグの再発防止)。

### v4+: 派生特徴(自動FE)ブロック

`file_version >= 4` の場合のみ、ヘッダ直後に挿入される。じっくりモードの自動特徴量
エンジニアリング(`_build_derived_recipe`)が採用した派生特徴のうち、**このモデルが実際に
使うもののみ**を書き出す(未使用の派生特徴は含めない)。

| 型 | 内容 |
|---|---|
| `u32` | `n_derived`(件数) |
| ×n_derived: `u8` | `op`(0=乗算`mul` / 1=二乗`sq` / 2=符号`sign`) |
| `str` | `name`(派生特徴の列名。例: `"x1*x2"`、`"x1^2"`、`"sign(x1)"`。以降`feat_cols`にこの名前で
  現れた場合、生列としてCSVから読むのではなくここで計算する) |
| `str` | `col_a`(ソース列1) |
| `f32`, `f32` | `col_a`のクリップ境界(`a_lo`, `a_hi`。学習時のXクリッピング境界をベイク) |
| `str` | `col_b`(ソース列2。`sq`/`sign`では空文字列) |
| `f32`, `f32` | `col_b`のクリップ境界(`b_lo`, `b_hi`) |

推論側の計算式(`clipped_source`で`col_a`/`col_b`を`[lo,hi]`にクリップしてから):

- `mul`: `a * b`
- `sq`: `a * a`
- `sign`: `sign(a)` (a>0→1, a<0→-1, a==0→0)

ソース列が欠損(NaN)なら派生特徴もNaN(→呼び出し側でmedian補完に委ねる)。

### モデル種別ペイロード

以降、`d = n_feat`。

#### type 0: linear

| 型 | 内容 |
|---|---|
| `f32[d]` | `mean`(中心化前の平均。StandardScaler相当) |
| `f32[d]` | `scale`(標準偏差。書き出し時に`max(scale, 1e-8)`済みで、読込側は素の除算のみ行う) |
| `f32[d]` | `coef`(Ridge係数) |
| `f32` | `intercept` |

予測式: `sum_i coef[i] * (x[i]-mean[i])/scale[i] + intercept`(倍精度で累積)。

#### type 1: lgbm

| 型 | 内容 |
|---|---|
| `u32` | `n_trees` |
| `u32` | (予約、未使用。常に0) |
| ×n_trees: `u32` | `n_leaves`(このツリーの葉数。1なら単葉=定数寄与で内部ノード0個) |
| `u32[ni]` | `split_feature`(`ni = n_leaves-1`個の内部ノードそれぞれの分岐特徴index) |
| `f32[ni]` | `threshold`(分岐しきい値) |
| `i32[ni]` | `left_child`(子ノードindex。**負値は葉**: `-(leaf_idx+1)`で符号化) |
| `i32[ni]` | `right_child`(同上) |
| `f32[n_leaves]` | `leaf_value` |

整合性検査: 全ノードで`split_feature[i] < n_feat`、`left/right_child[i]`は
`[-n_leaves, ni)`の範囲内であることを読込側が強制(範囲外はロード拒否)。推論ループにも
ノード訪問回数上限(`ni`回)があり、循環参照を作る細工ファイルでも無限ループしない。

予測式: 全木のleaf値の和(`rf`/`xt`は`average_output`フラグでleaf値を`1/木の本数`に
事前スケール済みのため、書き出し後は「和」ロジックのままLightGBMの「平均」出力と一致する)。

#### type 2: gp

| 型 | 内容 |
|---|---|
| `f32[d]` | `mean` |
| `f32[d]` | `scale` |
| `f32[d]` | `ls`(ARDカーネルの次元別length-scale) |
| `f32` | `sv`(signal variance) |
| `f32`, `f32` | `y_mean`, `y_std`(学習時のyの標準化パラメータ) |
| `u32` | `n_train`(学習点数。`GP_MAX_TRAIN=300`超はランダムサブサンプル済み) |
| `f32[n_train*d]` | `X_train`(標準化後の学習点、行優先) |
| `f32[n_train]` | `alpha`(カーネルリッジの双対係数) |

予測式: ARD-RBFカーネルで学習点との距離を計算し`alpha`で加重和、`y_mean`/`y_std`で逆標準化。

#### type 3: mlp

| 型 | 内容 |
|---|---|
| `f32[d]` | `mean` |
| `f32[d]` | `scale` |
| `u32` | `n_layers` |
| ×n_layers: `u32`, `u32`, `u8` | `n_in`, `n_out`, `act`(0=relu, 1=identity。最終層のみidentity) |
| `f32[n_in*n_out]` | `W`(sklearn準拠の行優先、`W[k*n_out+j]`) |
| `f32[n_out]` | `b` |

整合性検査: `layers[0].n_in == n_feat`、`layers[k].n_in == layers[k-1].n_out`の次元チェーンを
読込側が強制。層をまたぐ受け渡し時のみfloat32へ丸め、層内の積和は倍精度で累積。

#### type 4: linear_poly(多項式Ridge)

RobustScaler相当の中心化・スケーリング後、`PolynomialFeatures(degree=2, include_bias=False)`
相当の多項式展開(単項+二乗+ペア積)を経てRidge回帰する。行数≤`POLY_MAX_ROWS`(200)かつ
特徴数≤`POLY_MAX_FEATS`(8)の小データでのみ選ばれる。

| 型 | 内容 |
|---|---|
| `f32[d]` | `center` |
| `f32[d]` | `scale`(書き出し時に`max(scale,1e-8)`済み) |
| `u32` | `n_terms` |
| `i32[n_terms]`, `i32[n_terms]` | `term_a`, `term_b`(項ごとの特徴index対。**`term_b<0`は単項**
  `s[term_a]`を表す。`term_b>=0`かつ`term_a==term_b`は二乗、`term_a!=term_b`はペア積) |
| `f32[n_terms]` | `coef` |
| `f32` | `intercept` |

項の並び順は`_light.PolynomialFeatures.transform`と同一(`[単項(i昇順)]` +
`[i<=jの積(i昇順→j昇順、i==jが二乗)]`)にし、`model.coef_`の並びとそのまま対応させている。

予測式: `s[i] = (x[i]-center[i])/scale[i]`(**倍精度で計算してから最後に1回だけfloat32へ丸める**
— 2026-07-16修正、後述の既知の教訓を参照)、その後`sum_t coef[t] * (term_b[t]<0 ? s[term_a[t]] :
s[term_a[t]]*s[term_b[t]]) + intercept`(積演算自体も倍精度)。

#### type 5: blend(アンサンブル)

自身は`n_feat=0`で直接の特徴ベクトルを持たず、代わりに複数の**自己完結した入れ子`.treg`**を
メンバーとして持つ。

| 型 | 内容 |
|---|---|
| `u32` | `n_members` |
| ×n_members: `f32` | `weight`(このメンバーの加重和係数) |
| `u32` | `blob_len`(このメンバーの入れ子`.treg`バイト列長) |
| `byte[blob_len]` | メンバーの完全な`.treg`バイト列(ヘッダ`"TREG"`から共通テールまで丸ごと) |

各メンバーは「後処理なし」(`smear=1.0`, `y_clip`=無制限, `round_output=false`)かつ
`y_transform`は**自身の実測値**(外側のblendとは無関係に各メンバーが個別に逆変換を適用)で
書き出される。予測は`predictRow`をメンバーごとに再帰呼び出しして得た「実スケールの予測値」を
`weight`で加重和するだけ。最終的な`smear`/`y_clip`/`round_output`と`y_transform`(常に`none`固定
— 各メンバーで既に逆変換済みのため外側で再度適用すると二重適用になる)は、外側(blendモデル自身)
のY逆変換ブロック・予測後処理ブロックで一度だけ適用される。

**再帰深さ制限**: `depth > 1`(blendの中にblendが入れ子になっているケース)は読込側が拒否する
(現行のwriterはこの入れ子を生成しないため、1MB程度の細工された`.treg`でスタックオーバーフローを
起こす攻撃面を潰すための制限)。

### Y逆変換ブロック(v2以降で存在)

| 型 | 内容 |
|---|---|
| `u8` | `y_transform`(0=none, 1=log1p, 2=yeo_johnson) |
| `f32` | `lambda`(**`y_transform==2`のときのみ存在**) |

blendの外側モデルは常に`y_transform=none`固定で書かれる(前述のとおり二重適用防止)。

### 予測後処理ブロック(v3以降で存在)

| 型 | 内容 |
|---|---|
| `u8` | `round_output`(0/1) |
| `f32` | `smear`(log1p使用時のsmearing補正係数。それ以外は1.0) |
| `f32`, `f32` | `y_clip_lo`, `y_clip_hi`(学習時観測レンジ±マージン。範囲外は無制限を表す
  `±3.4e38`近傍のセンチネル値) |
| `u32` | `n_fc_clip`(Xクリッピング境界の列数。通常`n_feat`と同じ) |
| ×n_fc_clip: `f32`, `f32` | 各特徴列の`x_clip_lo`, `x_clip_hi` |

適用順序: `pred = raw_model_output` → `y_transform`逆変換 → `pred *= smear` →
`clip(pred, y_clip_lo, y_clip_hi)` → (`round_output`なら)半分丸め(0から遠い方向)。

### 共通テール(全バージョン共通)

| 型 | 内容 |
|---|---|
| `str` | `target_col` |
| `u32` | `n_fc`(feat_cols件数。ヘッダの`n_feat`と一致必須) |
| `str[n_fc]` | `feat_cols`(特徴量列名。派生特徴はここに派生特徴名で現れ、v4ブロックの`derived_idx`
  経由で計算対象と判定される) |
| `u32` | `n_fc`(再掲、medians件数として使用) |
| ×n_fc: `str`, `f64` | 列名, `median`(欠損値補完用。**ここのみ倍精度**) |

## EXEテール形式(exe版のみ)

配布exeは自身の実行バイナリ末尾に`.treg`を追記して自己完結させる:

```
[EXE本体バイト列] [.treg バイト列] [u64(リトルエンディアン) treg_size] [8バイト "TREG_EMB"]
```

読み込み側は`GetModuleFileNameW`で自身のパスを取得(`argv[0]`には依存しない)、ファイル末尾8
バイトが`"TREG_EMB"`と一致するかを検査し、その直前の`u64`(`_ftelli64`等64bit API使用。2GB超の
exeでも破綻しない)で`.treg`サイズを得てシークする。fseek/freadの戻り値も検査する。

## バージョン履歴

| version | 追加内容 |
|---|---|
| 1 | 基本形(ヘッダ+ペイロード+共通テール)。現在は書き出されない(旧`.treg`の後方互換読込のみ) |
| 2 | Y逆変換ブロック追加(`y_transform`/`lambda`) |
| 3 | 予測後処理ブロック追加(`round_output`/`smear`/`y_clip`/`x_clip`)。現行の書き出しの下限 |
| 4 | 派生特徴(自動FE)ブロック追加。**このモデルが実際に使う派生特徴が1件以上あるときのみ**
    v4になり、なければv3のまま書かれる(`_write_treg_stream`参照) |

読込側(`predict_native_v2.cpp`/`predict-core.js`)は`file_version <= 4`を受理し、5以上は
「このexeより新しいモデル形式です」として明示エラーで拒否する(将来のフォーマット変更時に
古い配布exeが黙って誤動作しないようにするため)。

## 既知の教訓(2026-07-16、type4/5フィクスチャ追加時に発覚)

`predict_linear_poly`の標準化ステップ`(x[i]-center[i])/scale[i]`を素朴に「float同士の減算→
float同士の除算」として実装すると、2回の丸めが積み重なる。JS版(`predict-core.js`
`predictLinearPoly`)はこの式全体を倍精度で計算してから結果を1回だけ`Float32Array`に格納する
(=1回だけ丸める)ため、極端な入力値(例: `x=1e6`、`center`/`scale`が1桁程度)を二乗する項では
両者の差が丸め後も無視できない絶対誤差(実測で~1e3〜1e4オーダー)として残った。
`predict_linear`/`predict_mlp`は同種の問題に2026-07の別修正で対応済みだったが、`linear_poly`は
未対応のまま残っていた。修正: 標準化の減算・除算、および項の積(`s[a]*s[b]`)を倍精度で計算し、
最後に1回だけ`float`へ丸める(JS側と丸めのタイミングを揃える)。

この教訓から、**新しいモデル種別・演算を追加する際は「どの段階で何回float32に丸めるか」を
JS/C++間で明示的に揃えること**を設計原則とする。テストランナー側でも、`round_output=true`だから
といって絶対誤差がゼロに近いとは限らない(丸めは1未満のノイズしか消さない)ため、
`round_output`の有無に関わらず期待値に応じた相対混合閾値(`max(1e-6, 1e-4*|期待値|)`)で判定する
方針に統一した(`web/js_predict_poc/run_matrix_test.js`)。

## 次回改定時の設計メモ

現行フォーマットには「ペイロード全長」の情報がなく、未知のバージョンや将来の拡張(例:
カテゴリエンコーダ収録)を読み飛ばす手段がない(`file_version`が読込側の対応範囲を超えると
即座に拒否するしかない)。**次回のフォーマット改定(例: v5でカテゴリエンコーダブロック追加等)
では、ヘッダ(`magic`+`file_version`+`model_type`+`n_feat`)の直後に`u32`の
「ペイロード全長(このモデル1個分、共通テールの終端までのバイト数)」を追加し、前方互換を
確保する**。これにより:

- 新しいバージョンの`.treg`を旧リーダーが読んだ場合、内容を解釈できなくても
  「ペイロード全長ぶんスキップして次に進む」判断が(blendの入れ子境界特定などで)容易になる。
- 破損ファイルの検査(読み取り位置がペイロード全長を超えていないか)がバージョン非依存で
  行える。
- blend入れ子ブロブの境界特定は現状`blob_len`(メンバー単位)で既に担保されているが、
  単体`.treg`ファイルとして外部から扱う際(ファイルサイズ検査等)にも同じ恩恵がある。

この変更は4点セット(train_bridge.py / predict_native_v2.cpp / predict-core.js /
predict_template.py+predict_template.html)+パリティフィクスチャの同時更新が必須(CLAUDE.md
ルール3)。既存v1〜v4ファイルとの共存は、フィールド追加時と同様「読込側のバージョン分岐に
1本追加」で対応可能(全長フィールド自体はv5以降にのみ存在させる設計を推奨)。
