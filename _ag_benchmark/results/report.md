# T-regressor バックエンド vs AutoGluon — 30問回帰ベンチマーク

## 比較プロトコル(公平性)

- **同一の train/test 分割**(固定シード=42, test=25%)を両ツールに与える。treg が書いた分割ファイルを AG も読むことで分割の同一性を厳密に保証。
- **中立採点**: 両ツールの test 予測を同一コード(純numpy R²/RMSE/MAE)で採点。treg の自己申告R²(OOF/val)は比較には使わず、別途「誠実性」の参考としてのみ記録。
- **同一の時間バジェット**: AG には各問題の treg *thorough* 実測学習時間を time_limit として与える(下限60s, preset=good_quality, Windows安定化のためRay不使用・dynamic_stacking無効)。「同じ時間でどちらが賢いか」の比較。
- 合成問題は既知の **ceiling_r2**(どんな学習器も原理的に超えられない test R²上限)を併記。

## ヘッドライン

- 対戦成立: **40問**(全40問)
- **勝敗(treg thorough vs AG, 差0.005以内は引分)**: AG勝ち **9** / treg勝ち **14** / 引分 **17**
- **平均 test R²**: treg quick=0.753, treg thorough=0.772, AG=0.762
- **AG − treg(thorough)の test R²差**: 中央値=-0.002, 平均=-0.010 (正ならAG優位)
- **平均学習時間**: treg quick=4.2s, treg thorough=109.6s, AG=120.7s
- **天井到達率(合成, test R²/ceiling)**: treg thorough=0.996, AG=0.974 (1.0で理論上限)
- **treg 自己申告の誠実性(self R² − test R²)**: 平均=-0.009 (正なら自己申告が楽観的)

## 全問結果(test集合・中立採点)

| # | dataset | src | family | n_tr/te | ceiling | treg quick | treg thorough | treg model | AG | AG model | 勝者 | AG−tt |
|--:|---|---|---|---|--:|--:|--:|---|--:|---|:--:|--:|
| 1 | pub_appliances | publ | public | 1500/500 | — | 0.184 | 0.252 | Blend (Ensemble) | 0.256 | WeightedEnsemble | tie | 0.004 |
| 2 | pub_automgp | publ | public | 298/100 | — | 0.880 | 0.916 | Blend (Ensemble) | 0.917 | WeightedEnsemble | tie | 0.001 |
| 3 | pub_concrete | publ | public | 772/258 | — | 0.909 | 0.928 | LightGBM | 0.944 | WeightedEnsemble | AG | 0.017 |
| 4 | pub_energy | publ | public | 576/192 | — | 0.998 | 0.998 | GaussianProcess  | 0.999 | WeightedEnsemble | tie | 0.001 |
| 5 | pub_forestfires | publ | public | 388/129 | — | -0.026 | -0.026 | GaussianProcess  | -0.012 | WeightedEnsemble | AG | 0.014 |
| 6 | pub_parkinsons | publ | public | 1500/500 | — | 0.923 | 0.965 | Blend (Ensemble) | 0.961 | WeightedEnsemble | tie | -0.003 |
| 7 | pub_powerplant | publ | public | 1500/500 | — | 0.946 | 0.955 | LightGBM | 0.954 | WeightedEnsemble | tie | -0.001 |
| 8 | pub_realestate | publ | public | 310/104 | — | 0.758 | 0.750 | Blend (Ensemble) | 0.761 | WeightedEnsemble | AG | 0.011 |
| 9 | pub_studentperf | publ | public | 487/162 | — | 0.176 | 0.209 | Blend (Ensemble) | 0.224 | WeightedEnsemble | AG | 0.015 |
| 10 | pub_superconduct | publ | public | 1500/500 | — | 0.845 | 0.877 | LightGBM | 0.881 | WeightedEnsemble | tie | 0.003 |
| 11 | real_abalone | real | real | 1500/500 | — | 0.465 | 0.497 | Blend (Ensemble) | 0.470 | WeightedEnsemble | treg | -0.027 |
| 12 | real_airfoil | real | real | 1127/376 | — | 0.906 | 0.961 | Blend (Ensemble) | 0.959 | WeightedEnsemble | tie | -0.001 |
| 13 | real_california | real | real | 1500/500 | — | 0.808 | 0.822 | Blend (Ensemble) | 0.817 | WeightedEnsemble | tie | -0.004 |
| 14 | real_cpu | real | real | 1500/500 | — | 0.864 | 0.961 | LightGBM | 0.963 | WeightedEnsemble | tie | 0.003 |
| 15 | real_diabetes | real | real | 332/110 | — | 0.459 | 0.517 | Blend (Ensemble) | 0.480 | WeightedEnsemble | treg | -0.038 |
| 16 | real_winequality | real | real | 1500/500 | — | 0.280 | 0.320 | Blend (Ensemble) | 0.409 | WeightedEnsemble | AG | 0.089 |
| 17 | categorical_high | synt | categorical | 450/150 | 0.975 | 0.944 | 0.954 | LightGBM | 0.952 | WeightedEnsemble | tie | -0.002 |
| 18 | categorical_interaction | synt | categorical | 450/150 | 0.929 | 0.928 | 0.929 | Linear (Ridge) | 0.911 | WeightedEnsemble | treg | -0.018 |
| 19 | categorical_low | synt | categorical | 450/150 | 0.975 | 0.967 | 0.967 | Linear (Ridge) | 0.965 | WeightedEnsemble | tie | -0.002 |
| 20 | collinear | synt | linear | 450/150 | 0.980 | 0.962 | 0.962 | Linear (Ridge) | 0.964 | WeightedEnsemble | tie | 0.002 |
| 21 | heteroscedastic | synt | linear | 525/175 | 0.822 | 0.812 | 0.787 | Blend (Ensemble) | 0.796 | WeightedEnsemble | AG | 0.009 |
| 22 | linear_clean | synt | linear | 600/200 | 0.997 | 0.996 | 0.995 | GaussianProcess  | 0.990 | WeightedEnsemble | tie | -0.005 |
| 23 | linear_highdim_sparse | synt | linear | 450/150 | 0.991 | 0.983 | 0.984 | Linear (Ridge) | 0.934 | WeightedEnsemble | treg | -0.049 |
| 24 | linear_noisy | synt | linear | 600/200 | 0.647 | 0.681 | 0.636 | GaussianProcess  | 0.628 | WeightedEnsemble | treg | -0.009 |
| 25 | mixed_messy | synt | mixed | 450/150 | 0.990 | 0.974 | 0.973 | Linear (Ridge) | 0.957 | WeightedEnsemble | treg | -0.016 |
| 26 | interaction_deep | synt | nonlinear | 600/200 | 0.902 | 0.886 | 0.898 | GaussianProcess  | 0.727 | WeightedEnsemble | treg | -0.171 |
| 27 | many_irrelevant | synt | nonlinear | 450/150 | 0.982 | 0.918 | 0.950 | Blend (Ensemble) | 0.955 | WeightedEnsemble | AG | 0.005 |
| 28 | monotonic_saturating | synt | nonlinear | 525/175 | 0.983 | 0.980 | 0.980 | GaussianProcess  | 0.979 | WeightedEnsemble | tie | -0.000 |
| 29 | multiplicative | synt | nonlinear | 525/175 | 0.965 | 0.949 | 0.951 | Linear (Ridge) | 0.888 | WeightedEnsemble | treg | -0.063 |
| 30 | nonlinear_interaction | synt | nonlinear | 675/225 | 0.877 | 0.792 | 0.870 | Blend (Ensemble) | 0.854 | WeightedEnsemble | treg | -0.016 |
| 31 | piecewise_steps | synt | nonlinear | 525/175 | 0.954 | 0.923 | 0.932 | LightGBM | 0.928 | WeightedEnsemble | tie | -0.004 |
| 32 | polynomial_deg3 | synt | nonlinear | 525/175 | 0.943 | 0.937 | 0.939 | GaussianProcess  | 0.929 | WeightedEnsemble | treg | -0.010 |
| 33 | radial_rbf | synt | nonlinear | 450/150 | 0.951 | 0.913 | 0.928 | GaussianProcess  | 0.937 | WeightedEnsemble | AG | 0.009 |
| 34 | trig_smooth | synt | nonlinear | 525/175 | 0.953 | 0.934 | 0.936 | GaussianProcess  | 0.932 | WeightedEnsemble | tie | -0.004 |
| 35 | xor_sign | synt | nonlinear | 525/175 | 0.961 | 0.849 | 0.949 | LightGBM | 0.913 | WeightedEnsemble | treg | -0.035 |
| 36 | count_poisson | synt | pathology | 525/175 | 0.526 | 0.570 | 0.590 | Linear (Ridge) | 0.625 | WeightedEnsemble | AG | 0.035 |
| 37 | outlier_contaminated | synt | pathology | 450/150 | 0.157 | 0.170 | 0.184 | Linear (Ridge) | 0.181 | WeightedEnsemble | tie | -0.003 |
| 38 | pure_noise | synt | pathology | 450/150 | 0.000 | -0.064 | -0.022 | LightGBM | -0.032 | WeightedEnsemble | treg | -0.010 |
| 39 | skewed_target | synt | pathology | 450/150 | 0.861 | 0.829 | 0.837 | Blend (Ensemble) | 0.816 | WeightedEnsemble | treg | -0.021 |
| 40 | small_n | synt | pathology | 30/10 | 0.976 | 0.925 | 0.885 | Blend (Ensemble) | 0.780 | WeightedEnsemble | treg | -0.105 |

## ソース別 平均 test R²

| source | n | treg quick | treg thorough | AG |
|---|--:|--:|--:|--:|
| synthetic | 24 | 0.823 | 0.833 | 0.813 |
| real | 6 | 0.630 | 0.680 | 0.683 |

## treg が最も離された問題(AG−tt 上位)

- **real_winequality** (real): tt=0.320 vs AG=0.409 (差 0.089) — 赤ワイン品質スコア回帰(n=1599)
- **count_poisson** (pathology): tt=0.590 vs AG=0.625 (差 0.035) — ポアソン計数(整数・歪み)
- **pub_concrete** (public): tt=0.928 vs AG=0.944 (差 0.017) — コンクリート圧縮強度(UCI 165, n=1030)。CTR23/AMLB系論文で頻出の定番
- **pub_studentperf** (public): tt=0.209 vs AG=0.224 (差 0.015) — 生徒の最終成績(UCI 320, n=395)。G1/G2(中間成績)は最終成績と強相関のため除外し純粋な特徴から予測
- **pub_forestfires** (public): tt=-0.026 vs AG=-0.012 (差 0.014) — 森林火災の焼失面積(UCI 162, n=517)。極端な右裾(0が大半)、頑健性の試金石

## treg が競り勝った/並んだ問題(AG−tt 下位)

- **interaction_deep** (nonlinear): tt=0.898 vs AG=0.727 (差 -0.171) — 3方向交互作用(加法モデルに極めて不利)
- **small_n** (pathology): tt=0.885 vs AG=0.780 (差 -0.105) — 小標本(n=40、過学習リスク)
- **multiplicative** (nonlinear): tt=0.951 vs AG=0.888 (差 -0.063) — 純交互作用(主効果ゼロ)。線形に厳しい
- **linear_highdim_sparse** (linear): tt=0.984 vs AG=0.934 (差 -0.049) — 高次元(60特徴)中6本のみ有効。特徴選抜
- **real_diabetes** (real): tt=0.517 vs AG=0.480 (差 -0.038) — sklearn同梱。糖尿病進行度(n=442,10特徴)
