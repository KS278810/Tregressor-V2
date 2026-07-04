# _light.py の各コンポーネントが sklearn/scipy と数値一致することを検証する。
# Phase3(sklearn軽量置換)の安全性ゲート。train_bridge へ配線する前に必ず通す。
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import _light as L

FAIL = []
def chk(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}: {name}" + (f"  ({detail})" if detail else ""))
    if not cond: FAIL.append(name)

rng = np.random.RandomState(0)

# ── metrics ──
from sklearn.metrics import r2_score as sk_r2, mean_squared_error as sk_mse, mean_absolute_error as sk_mae
yt = rng.randn(200); yp = yt + rng.randn(200)*0.3
chk("r2_score", abs(L.r2_score(yt,yp)-sk_r2(yt,yp))<1e-12, f"{L.r2_score(yt,yp):.6f}")
chk("mse", abs(L.mean_squared_error(yt,yp)-sk_mse(yt,yp))<1e-12)
chk("mae", abs(L.mean_absolute_error(yt,yp)-sk_mae(yt,yp))<1e-12)

# ── scalers (属性名も一致) ──
from sklearn.preprocessing import StandardScaler as SkStd, RobustScaler as SkRob
X = rng.randn(150,5)*3 + 2
ls = L.StandardScaler().fit(X); ss = SkStd().fit(X)
chk("StandardScaler.mean_", np.allclose(ls.mean_, ss.mean_))
chk("StandardScaler.scale_", np.allclose(ls.scale_, ss.scale_))
chk("StandardScaler.transform", np.allclose(ls.transform(X), ss.transform(X)))
lr = L.RobustScaler().fit(X); sr = SkRob().fit(X)
chk("RobustScaler.center_", np.allclose(lr.center_, sr.center_))
chk("RobustScaler.scale_", np.allclose(lr.scale_, sr.scale_))
chk("RobustScaler.transform", np.allclose(lr.transform(X), sr.transform(X)))

# ── PolynomialFeatures (列順・値・名前) ──
from sklearn.preprocessing import PolynomialFeatures as SkPoly
Xp = rng.randn(30,4)
lp = L.PolynomialFeatures(degree=2, include_bias=False)
spf = SkPoly(degree=2, include_bias=False)
LT = lp.fit_transform(Xp); ST = spf.fit_transform(Xp)
chk("PolynomialFeatures.shape", LT.shape==ST.shape, f"{LT.shape} vs {ST.shape}")
chk("PolynomialFeatures.values", np.allclose(LT, ST))
ln = list(lp.get_feature_names_out([f"c{i}" for i in range(4)]))
chk("PolynomialFeatures.names数", len(ln)==ST.shape[1])

# ── RidgeCV (外部テストR²一致) ──
from sklearn.linear_model import RidgeCV as SkRidge
alphas=[0.001,0.01,0.1,1.,10.,100.,1000.]
Xtr=rng.randn(200,6); ytr=Xtr@np.array([2,-1,0.5,0,3,-2.])+rng.randn(200)*0.5
Xte=rng.randn(100,6); yte=Xte@np.array([2,-1,0.5,0,3,-2.])+rng.randn(100)*0.5
lm=L.RidgeCV(alphas=alphas).fit(Xtr,ytr); sm=SkRidge(alphas=alphas).fit(Xtr,ytr)
chk("RidgeCV 外部R²一致", abs(L.r2_score(yte,lm.predict(Xte))-sk_r2(yte,sm.predict(Xte)))<0.01,
    f"自前={L.r2_score(yte,lm.predict(Xte)):.4f} sk={sk_r2(yte,sm.predict(Xte)):.4f}")
chk("RidgeCV coef_ 属性存在", hasattr(lm,'coef_') and hasattr(lm,'intercept_') and hasattr(lm,'alpha_'))

# ── PowerTransformer (yeo-johnson: λ・変換・逆変換) ──
from sklearn.preprocessing import PowerTransformer as SkPT
for name, x in [("skew+", np.abs(rng.randn(300))**1.5 + 0.1),
                ("mixed", rng.randn(300)*2)]:
    lt=L.PowerTransformer(method='yeo-johnson',standardize=False).fit(x.reshape(-1,1))
    st=SkPT(method='yeo-johnson',standardize=False).fit(x.reshape(-1,1))
    dlam=abs(float(lt.lambdas_[0])-float(st.lambdas_[0]))
    chk(f"PowerTransformer λ一致[{name}]", dlam<0.05, f"自前={float(lt.lambdas_[0]):.3f} sk={float(st.lambdas_[0]):.3f}")
    # 変換→逆変換で復元
    xt=lt.transform(x.reshape(-1,1))
    xi=lt.inverse_transform(xt).ravel()
    chk(f"PowerTransformer 逆変換往復[{name}]", np.allclose(xi, x, atol=1e-6))
    # sklearn変換との一致（同一λを注入して比較: 変換式の一致確認）
    lt2=L.PowerTransformer(method='yeo-johnson',standardize=False); lt2.lambdas_=st.lambdas_.copy()
    chk(f"PowerTransformer 変換式一致[{name}]", np.allclose(lt2.transform(x.reshape(-1,1)).ravel(),
        st.transform(x.reshape(-1,1)).ravel(), atol=1e-6))

# ── skew / nnls ──
from scipy.stats import skew as sp_skew
from scipy.optimize import nnls as sp_nnls
for x in [rng.randn(100), np.abs(rng.randn(100))**2]:
    chk("skew一致", abs(L.skew(x)-float(sp_skew(x)))<1e-9)
A=np.abs(rng.randn(80,5)); b=A@np.array([1.,0,2,0,0.5])+rng.randn(80)*0.1
xl,_=L.nnls(A,b); xs,_=sp_nnls(A,b)
chk("nnls fitR²一致", abs(L.r2_score(b,A@xl)-L.r2_score(b,A@xs))<1e-6)
chk("nnls 非負", (xl>=0).all())

# ── permutation_importance (符号の整合: 効く特徴が上位) ──
class Lin:
    def __init__(s,w): s.w=w
    def predict(s,X): return X@s.w
est=Lin(np.array([5.,0,0,0.,0])); Xi=rng.randn(200,5)
res=L.permutation_importance(est, Xi, Xi@est.w, n_repeats=5)
chk("permutation 重要度: 特徴0が最大", int(np.argmax(res.importances_mean))==0)

print()
if FAIL:
    print(f"FAIL {len(FAIL)}: {FAIL}"); sys.exit(1)
print("ALL PASS: _light は sklearn/scipy と数値一致")
