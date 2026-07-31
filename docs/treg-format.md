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
[v5+: カテゴリエンコーダ(cat_encoders)ブロック]
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
| +4 | `u8` | `file_version`(現行の書き出しは3〜7。読み込み側は1〜7を受理し、8以上は明示エラーで拒否) |
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

### v5+: カテゴリエンコーダ(cat_encoders)ブロック

`file_version >= 5` の場合のみ、v4派生特徴ブロックの直後に挿入される(v4ブロック自体は
`file_version >= 4`なら常に存在するため、v5〜v7モデルは必ずv4ブロックとこのブロックの両方を
持つ。`_write_treg_stream`の
`file_version = 7 if used_composite else (6 if used_dt else (5 if used_cat else (4 if used_derived else 3)))`
参照)。ブロック自体は`file_version >= 5`で読む点はv6/v7でも変わらない(method==2の
datetime_partsやmethod==3のcomposite_targetエントリが1件以上混在する場合のみ、writerが
file_versionをそれぞれ6・7に上げる)。

精度レバー4(カテゴリ処理刷新、2026-07-16)により、`train_bridge._prepare_categoricals`が
学習前にCSVの非数値列を次の判定順で4方式に振り分ける(**上から順に判定し、最初にヒットした
方式が採用される**。datetime_partsは数値化の直後・カーディナリティ判定より前に評価するため、
低カーディナリティの日時列であってもone-hotではなくdatetime_partsが優先される):

1. **数値化**: `pd.to_numeric`で90%以上の値が数値変換できる列はそのまま数値列として扱う
   (`cat_encoders`には現れない)。
2. **datetime_parts**(2026-07第3弾・真因④対策。独自の日時正規表現
   `^(\d{4})[-/](\d{2})[-/](\d{2})(?:[ T]?(\d{2}):(\d{2})(?::(\d{2}))?)?$`に90%以上マッチする
   場合): target非依存でグローバル適用(one-hotと同じくfold-aware refit不要)。1source_col
   につき`hour`/`dow`/`month`/`epoch_days`の4つの数値派生列(`{col}__hour`等)を生成する。
   年前置き(ISO/スラッシュ)のみ対応、US式`MM/DD/YYYY`等は対象外(既知の制限)。区切りなし
   連結(`"2016-01-1313:50:00"`、UCI appliancesデータセットで実際に確認された壊れた
   フォーマット)も固定幅2桁のため曖昧にならず対応する。カーディナリティ判定より先に評価する
   ため、低カーディナリティ日時列(3種類の日付を持つグループラベル等)もdatetime_parts側に
   ルーティングされる(意図的なトレードオフ)。
3. **one-hot**(カーディナリティ≤10): クラスごとに`col=="A"`のような indicator 列(0.0/1.0)を
   生成し、`feat_cols`にはこの生成列名が現れる(target非依存で安全にグローバル適用可能)。
   未知カテゴリ・学習時に見なかった欠損は「全クラスでマッチなし」= 実質0として扱われる。
4. **target encoding**(10 < カーディナリティ ≤ 行数の50%): 元の列名のまま、カテゴリ文字列を
   スムージング付き平均目的変数値に置き換える(fold内fitでOOFリークを防ぐ。カーディナリティが
   行数の50%を超える列は列ごと除外し、モデルには一切現れない)。

上記は非数値列のみが対象だが、**合成キー(composite_target、2026-07第3弾・真因②対策)**は
別経路で数値列を走査する(`train_bridge._detect_numeric_composite_key`)。反復測定データの
被験者代理キー(例: age+sexの組み合わせが実質的な被験者ID)のような「複数の低カーディナリティ
数値列の組み合わせ」を検出し、target encodingの1変種として1本の新列を追加する
(元の数値列は削除しない。単体でも有効な連続特徴のため、one-hotとは異なり置き換えない)。
検出条件: 各候補列は単体でnunique/行数比が5%以下かつ全値が整数相当であること、かつ
候補列すべてを組み合わせたグループ数/行数比が50%以下であること(`CAT_DROP_CARD_FRAC`と
同じ思想)。診断実験(`_ag_benchmark/diagnose_parkinsons_subject_encoding.py`)でpub_parkinsons
に対しLightGBM単体+0.026・GP単体+0.023の改善を実測済み。

`feat_cols`に現れる名前が「生のCSV列名そのもの」なのか「カテゴリエンコーダの生成列/変換後の
列」なのかは、この`cat_encoders`ブロックの`feature_name`と照合して判定する(v4派生特徴の
`derived_idx`と同じ判定パターン: まず派生特徴か確認 → 次にカテゴリエンコーダか確認 →
どちらでもなければ生のCSV列として読む)。

| 型 | 内容 |
|---|---|
| `u32` | `n_cat`(件数。**このモデルが実際に使うカテゴリ特徴のみ**を書き出す。未使用分は含めない) |
| ×n_cat: `u8` | `method`(0=one-hot / 1=target encoding / 2=datetime_parts、v6以降のみ出現 /
  3=composite_target、v7以降のみ出現) |
| `str` | `feature_name`(`feat_cols`に現れる名前。one-hotなら生成indicator列名`"col==クラス値"`、
  targetなら元のCSV列名そのもの、datetime_partsなら生成列名`"col__hour"`等、
  composite_targetなら生成列名`"__numkey__col1_col2"`等) |
| `str` | `source_col`(one-hot/target/datetime_partsは元のCSV列名1つ。**composite_targetのみ
  例外**で、`NUMKEY_SEP`(`\x1f`、ASCII unit separator)で連結した複数のCSV列名になる。
  one-hotではこの列の生文字列値と`class_value`を比較し、targetでもこの列の生文字列値を
  マップで引き、datetime_partsでもこの列の生文字列値を日時としてパースし、composite_target
  では連結された各列を分割して合成キーを再構築する) |
| method==0(one-hot)のみ: `str` | `class_value`(この1エントリが担当する比較対象クラス値) |
| method==1(target)またはmethod==3(composite_target)のみ: `u32` | `n_map`(カテゴリ→値
  マップの件数。両methodでペイロード形式は完全に同一) |
| ×n_map: `str`, `f32` | `class_str`, `value`(スムージング済みtarget encoding値。
  composite_targetでは`class_str`が合成キー、例えば`"65\x1f0"`) |
| method==1またはmethod==3のみ: `f32` | `default`(未知カテゴリ・学習時に見なかった欠損時の
  フォールバック値。学習側のグローバル平均目的変数値) |
| method==2(datetime_parts)のみ: `u8` | `part_id`(0=hour / 1=dow / 2=month / 3=epoch_days) |

推論側の計算(生CSV列の文字列値をそのまま使う。学習時にNaN/空セルは`"__NaN__"`という
サンチネル文字列に正規化してからカテゴリ照合しているため、推論側でも欠損セルは同じ
サンチネル文字列として扱う):

- **one-hot**: `source_col`の生文字列値が`class_value`と等しければ`1.0`、そうでなければ`0.0`
  (未知カテゴリ・学習時未見の欠損は必ずどのクラスとも一致しないため実質全ゼロになる)。
- **target**: `source_col`の生文字列値を`class_str → value`のマップで引く。マップに無ければ
  `default`。
- **datetime_parts**(v6): `source_col`の生文字列値を`_prepare_categoricals`と同一の日時
  正規表現でパースし、`part_id`に応じて`hour`(0-23)/`dow`(0=Mon..6=Sun)/`month`(1-12)/
  `epoch_days`(1970-01-01起点の整数日数、Howard Hinnantの`days_from_civil`で算出)を返す。
  **専用のdefaultフィールドは持たない**: パース失敗(学習時に見なかった不正フォーマット等)
  時はNaNを返し、全feat_col共通の既存の中央値フォールバック(`medians`、共通テール参照)に
  委ねる設計にした(target_defaultのような専用フィールドを増やしても4実装間で同期すべき
  箇所が増えるだけでメリットがないため)。この判定関数(検出率90%閾値の計算と抽出の両方で
  使う「唯一の真実の判定」)は`train_bridge._parse_datetime_parts`/
  `predict_native_v2.cpp parse_datetime_parts`/`predict-core.js parseDatetimeParts`
  (+`predict_template.html`インライン複製)/`predict_template.py _parse_datetime_parts`の
  4箇所に一字一句同じロジックで移植されている(参照日テストで全一致確認済み)。
- **composite_target**(v7): `source_col`を`NUMKEY_SEP`(`\x1f`)で分割して複数の生CSV列名を
  復元し、各列の生文字列値を数値としてパース(既存の数値パーサを再利用、
  `predict_native_v2.cpp`の`parse_numeric_field`等)した上で、整数相当の値のみを対象に
  正規化(`train_bridge._canon_numeric_key_part`: 整数相当でない値・非有限値は
  `"__NaN__"`。指数表記等の言語間フォーマット差異を避けるため整数相当の値のみ
  サポートする設計)し、同じ区切り文字で再連結した合成キーで`class_str → value`の
  マップを引く。マップに無ければ`default`。datetime_partsと同様、**専用のdefault
  フィールドは1つだけ持つ**(method==1と共有、target同様の意味論)。この判定関数群
  (`_canon_numeric_key_part`+合成キー構築)は`train_bridge.py`/
  `predict_native_v2.cpp`(`canon_numeric_key_part`/`build_composite_key`)/
  `predict-core.js`(`canonNumericKeyPart`/`buildCompositeKey`、
  +`predict_template.html`インライン複製)/`predict_template.py`
  (`_canon_numeric_key_part`/`_build_composite_key`)の4箇所に移植されている。

派生特徴の`col_a`/`col_b`がカテゴリエンコーダの生成列(one-hot indicator名やtarget-encoding
後の元列名)を指すケースもある(例: `x1*grade==B`のようなFE由来の交差項)。この場合、
派生特徴の計算に先立って`col_a`/`col_b`をこのカテゴリエンコーダ経由で1段解決してから
掛け算・クリップ等を行う(3実装とも同一の名前解決優先順位: 派生特徴 → カテゴリエンコーダ →
生CSV列、`predict_native_v2.cpp`の`resolve_named`/`predict-core.js`の`resolveNamed`参照)。

列欠損警告(学習時の必須列がCSVに無い場合の警告)も、feat名がカテゴリエンコーダの生成列を
指す場合は`source_col`まで1段解決してから判定する(`predict_native_v2.cpp`の
`raw_sources_for`/`collect_required_raw_columns`、`predict-core.js`側は呼び出し元
`predict_template.html`の`rawSourcesFor`/`rawRequiredColumns`、Python版は
`predict_template.py`の`_to_raw_required`参照)。**composite_targetのみ1つのfeat名から
複数の生CSV列名(`source_col`をNUMKEY_SEPで分割したもの)が展開される**点が他のmethodと
異なる(v7で追加された唯一の非1:1解決パターン)。

blend(type 5)のメンバーは各々が自己完結した`.treg`として個別に`cat_encoders`を持つ
(外側のblendラッパー自身は`feat_cols=[]`のため`used_cat`が常に空になり、ラッパー自身の
`file_version`はv5に上がらず3のまま書かれる。v4派生特徴と同じ挙動。カテゴリ特徴は
各メンバーの入れ子`.treg`側にそれぞれ記録される)。

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

### 既知の制約: 書き出しEXEは常に無署名

`native_dist/predict_native.exe`自体は署名していない(Phase9時点で有償コード署名証明書は
未導入)。加えて、上記の「EXE本体バイト列の末尾に`.treg`を生バイトで追記する」方式は、
仮に将来ベースの`predict_native.exe`だけを署名したとしても、Authenticodeの検証対象が
PEファイルの構造化領域(証明書テーブルが指すオフセットまで)であるのに対し、末尾追記は
その外側にバイト列を継ぎ足す行為そのものであり、署名済みバイナリへの追記は「改変」として
署名検証を無効化する(≒その場で自己署名を破壊する)。つまり**「ベースexeを署名する」と
「モデルを追記して自己完結exeにする」は設計上両立しない**。署名を保ちたい場合は追記方式
自体を作り直す必要がある(例: 証明書テーブルより前の領域にモデルを埋め込む、外部リソース
ファイルとして分離する等)。現状はこのトレードオフを受け入れ、単体exe配布の利便性を優先して
無署名のまま運用する方針(README.mdの「書き出しEXEとウイルス対策ソフト」節、AV誤検知の
運用対処を参照)。

## バージョン履歴

| version | 追加内容 |
|---|---|
| 1 | 基本形(ヘッダ+ペイロード+共通テール)。現在は書き出されない(旧`.treg`の後方互換読込のみ) |
| 2 | Y逆変換ブロック追加(`y_transform`/`lambda`) |
| 3 | 予測後処理ブロック追加(`round_output`/`smear`/`y_clip`/`x_clip`)。現行の書き出しの下限 |
| 4 | 派生特徴(自動FE)ブロック追加。**このモデルが実際に使う派生特徴が1件以上あるときのみ**
    v4になり、なければv3のまま書かれる(`_write_treg_stream`参照) |
| 5 | カテゴリエンコーダ(`cat_encoders`)ブロック追加(2026-07-16、精度レバー4「カテゴリ処理刷新」)。
    **このモデルが実際に使うカテゴリ特徴(one-hot/target encoding)が1件以上あるときのみ**v5に
    なり、なければv4以下のまま書かれる(v4派生特徴と同じ判定パターン) |
| 6 | `cat_encoders`にmethod=2(`datetime_parts`)追加(2026-07第3弾、真因④対策「datetime列が
    ID列扱いで破棄される」の解消)。ブロック自体はv5のまま(構造変更なし)で、
    **このモデルが実際に使うdatetime_partsエントリが1件以上あるときのみ**v6になり、
    なければv5以下のまま書かれる(v4/v5と同じ判定パターン) |
| 7 | `cat_encoders`にmethod=3(`composite_target`)追加(2026-07第3弾、真因②対策「反復測定
    データの被験者代理キーのような複数低カーディナリティ数値列の組み合わせが未活用」の
    解消)。ペイロード形式はmethod=1(target)と完全に同一で、`source_col`の意味論のみ
    (単一CSV列名→NUMKEY_SEP連結の複数CSV列名)が異なる。**このモデルが実際に使う
    composite_targetエントリが1件以上あるときのみ**v7になり、なければv6以下のまま
    書かれる(v4/v5/v6と同じ判定パターン) |

読込側(`predict_native_v2.cpp`/`predict-core.js`)は`file_version <= 7`を受理し、8以上は
「このexeより新しいモデル形式です」として明示エラーで拒否する(将来のフォーマット変更時に
古い配布exeが黙って誤動作しないようにするため)。v5以下の**旧リーダーはv6ファイルを拒否する**
(`file_version > 5`のチェックのままなので、v6を読ませようとすると同じ「新しい形式です」
エラーになる。これにより新旧の配布物が混在しても誤動作なく安全側に倒れる)。

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

現行フォーマットには「ペイロード全長」の情報がなく、未知のバージョンや将来の拡張を読み飛ばす
手段がない(`file_version`が読込側の対応範囲を超えると即座に拒否するしかない)。v5(カテゴリ
エンコーダブロック追加)・v6(datetime_parts追加)・v7(composite_target追加)はいずれも
この制約の範囲内(既存の「バージョン分岐に1本追加」パターン)で対応できた。**v6・v7実装時
とも本ペイロード長u32の追加を検討したが、あえて見送った**: 効果は「未来の未知バージョンを
旧リーダーが安全にスキップする」という前方互換のみで、各バージョン自体は従来通り次の
バージョンを明示エラーで拒否するため恩恵が発生せず、一方で4実装全てにフィールド追加の手間と
リスクが生じる非対称なトレードオードだったため。
**次回のフォーマット改定(v8以降)で改めて追加が必要になった時点**で、ヘッダ(`magic`+
`file_version`+`model_type`+`n_feat`)の直後に`u32`の「ペイロード全長(このモデル1個分、
共通テールの終端までのバイト数)」を追加し、前方互換を確保することを推奨する。これにより:

- 新しいバージョンの`.treg`を旧リーダーが読んだ場合、内容を解釈できなくても
  「ペイロード全長ぶんスキップして次に進む」判断が(blendの入れ子境界特定などで)容易になる。
- 破損ファイルの検査(読み取り位置がペイロード全長を超えていないか)がバージョン非依存で
  行える。
- blend入れ子ブロブの境界特定は現状`blob_len`(メンバー単位)で既に担保されているが、
  単体`.treg`ファイルとして外部から扱う際(ファイルサイズ検査等)にも同じ恩恵がある。

この変更は4点セット(train_bridge.py / predict_native_v2.cpp / predict-core.js /
predict_template.py+predict_template.html)+パリティフィクスチャの同時更新が必須(CLAUDE.md
ルール3)。既存v1〜v7ファイルとの共存は、フィールド追加時と同様「読込側のバージョン分岐に
1本追加」で対応可能(全長フィールド自体は導入バージョン以降にのみ存在させる設計を推奨)。
