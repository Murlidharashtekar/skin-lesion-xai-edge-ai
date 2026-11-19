from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
CL=["akiec","bcc","bkl","df","mel","nv","vasc"]; S=Path("phase2_research/outputs/seeds")
def f1m(y,p):
    cm=np.bincount(y*7+p,minlength=49).reshape(7,7); tp=np.diag(cm).astype(float); d=cm.sum(0)+cm.sum(1)
    return float(np.divide(2*tp,d,out=np.zeros(7),where=d>0).mean())
def ece(P,y,b=15):
    c=P.max(1); ok=(P.argmax(1)==y); e=0.0; ed=np.linspace(0,1,b+1)
    for lo,hi in zip(ed[:-1],ed[1:]):
        m=(c>lo)&(c<=hi)
        if m.any(): e+=m.mean()*abs(ok[m].mean()-c[m].mean())
    return e
def nll(P,y): return float(-np.log(np.clip(P[np.arange(len(y)),y],1e-6,1)).mean())
def ts(P,T):
    z=np.log(np.clip(P,1e-4,1))/T; z-=z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
def load(tag,s):
    d=pd.read_csv(S/f"{tag}_{s}.csv"); P=d[[f"p_{c}" for c in CL]].values; P/=P.sum(1,keepdims=True); return d.y_true.values,P
a=pd.read_csv("phase2_research/outputs/all_test_predictions.csv")
rows=[]
for m,t in (("E1","E1_effnet_b3"),("E3","E3_fusion")):
    for sd in (42,43,44):
        tag=t if sd==42 else f"{t}_s{sd}"
        if (S/f"{tag}_val.csv").exists():
            yv,Pv=load(tag,"val"); yt,Pt=load(tag,"test")
            T=minimize_scalar(lambda T:nll(ts(Pv,T),yv),bounds=(0.3,5),method="bounded").x
            rows.append(dict(model=m,seed=sd,test_f1=f1m(yt,Pt.argmax(1)),ece=ece(Pt,yt),ece_ts=ece(ts(Pt,T),yt),T=T,nll=nll(Pt,yt),nll_ts=nll(ts(Pt,T),yt)))
        elif sd==42:
            d=a[a.run==t]; rows.append(dict(model=m,seed=42,test_f1=f1m(d.y_true.values,d.y_pred.values)))
            print("note:",m,"seed 42 validation export missing; calibration skipped for it")
r=pd.DataFrame(rows); pd.set_option("display.width",200)
print(r.round(4).to_string(index=False))
print("\nTest macro-F1 across seeds\n",r.groupby("model").test_f1.agg(["mean","std","count"]).round(4))
cal=r.dropna(subset=["T"]); print("\nCalibration means (seeds with val export)\n",cal.groupby("model")[["ece","ece_ts","nll","nll_ts","T"]].mean().round(4))
r.to_csv("phase2_research/outputs/eval/seed_pooled.csv",index=False)
