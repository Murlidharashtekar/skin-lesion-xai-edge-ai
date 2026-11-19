from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
CL=["akiec","bcc","bkl","df","mel","nv","vasc"]; C2I={c:i for i,c in enumerate(CL)}
S=Path("phase2_research/outputs/seeds")
te=pd.read_csv("data/splits_v2/test.csv"); y=te.dx.map(C2I).values
P={}
for k,t in (("E1","E1_effnet_b3"),("E2_e30","E2_deit_small_e30"),("E3","E3_fusion")):
    f=S/f"{t}_test.csv"
    if not f.exists(): print("missing",f); continue
    d=pd.read_csv(f); assert (d.y_true.values==y).all(), f"label order mismatch in {t}"
    P[k]=d[[f"p_{c}" for c in CL]].values.argmax(1)
def f1m(y,p):
    cm=np.bincount(y*7+p,minlength=49).reshape(7,7); tp=np.diag(cm).astype(float); d=cm.sum(0)+cm.sum(1)
    return float(np.divide(2*tp,d,out=np.zeros(7),where=d>0).mean())
groups=list(te.groupby("lesion_id").indices.values()); rng=np.random.default_rng(0); B=2000
idxs=[np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]) for _ in range(B)]
boot={k:np.array([f1m(y[i],p[i]) for i in idxs]) for k,p in P.items()}
print("seed-42 test macro-F1 with lesion-bootstrap 95% CI")
for k,p in P.items(): print(f"  {k}: {f1m(y,p):.4f} [{np.percentile(boot[k],2.5):.4f}, {np.percentile(boot[k],97.5):.4f}]")
print("\npaired differences (seed 42 only)")
for a,b in (("E2_e30","E1"),("E3","E1"),("E3","E2_e30")):
    if a in P and b in P:
        d=boot[a]-boot[b]; ca,cb=P[a]==y,P[b]==y
        n10,n01=int((ca&~cb).sum()),int((~ca&cb).sum()); pv=binomtest(n10,n10+n01,0.5).pvalue if n10+n01 else 1.0
        print(f"  {a} - {b}: dF1 {f1m(y,P[a])-f1m(y,P[b]):+.4f} CI [{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}] | McNemar {n10}/{n01} p={pv:.3f}")
