import base64, io, zlib, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

ROOT = Path("phase2_research/outputs")
OUT = ROOT / "eval"; OUT.mkdir(exist_ok=True, parents=True)
CL = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
REF = {"E1_effnet_b3": 0.6611, "E2_deit_small": 0.6327, "E3_fusion": 0.6908, "E4_effnet_b3_cw": 0.6478}

lines = [l.strip() for l in (ROOT / "blob_clean.txt").read_text().splitlines() if l.strip() and not l.startswith("---")]
blob = "".join(lines)
print("blob length:", len(blob), "(expected 101192)")
df = pd.read_csv(io.StringIO(zlib.decompress(base64.b64decode(blob)).decode()))
df.to_csv(ROOT / "all_test_predictions.csv", index=False)
runs = list(REF)
P = {r: df[df.run == r].reset_index(drop=True) for r in runs}
base = P[runs[0]]
for r in runs:
    assert len(P[r]) == 1431 and (P[r].image_id == base.image_id).all(), f"row mismatch in {r}"

def cm_of(y, p): return np.bincount(y * 7 + p, minlength=49).reshape(7, 7)
def prf(cm):
    tp = np.diag(cm).astype(float); rs = cm.sum(1); cs = cm.sum(0)
    pr = np.divide(tp, cs, out=np.zeros(7), where=cs > 0)
    rc = np.divide(tp, rs, out=np.zeros(7), where=rs > 0)
    f1 = np.divide(2 * tp, rs + cs, out=np.zeros(7), where=(rs + cs) > 0)
    return pr, rc, f1
def summ(y, p):
    cm = cm_of(y, p); pr, rc, f1 = prf(cm)
    return dict(accuracy=np.trace(cm) / cm.sum(), macro_precision=pr.mean(), macro_recall=rc.mean(),
                macro_f1=f1.mean(), weighted_f1=(f1 * cm.sum(1)).sum() / cm.sum())
def ece(prob, y, bins=15):
    conf = prob.max(1); pred = prob.argmax(1); ok = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return e

y = base.y_true.values
Y = {r: P[r].y_true.values for r in runs}
pred = {r: P[r].y_pred.values for r in runs}
prob = {r: P[r][[f"p_{c}" for c in CL]].values for r in runs}
prob = {r: v / v.sum(1, keepdims=True) for r, v in prob.items()}

# 1. point estimates + sanity check against Kaggle results.json
rows = []
for r in runs:
    s = summ(y, pred[r]); s["run"] = r
    s["ece"] = ece(prob[r], y)
    s["brier"] = float(((prob[r] - np.eye(7)[y]) ** 2).sum(1).mean())
    s["auc_ovr_macro"] = roc_auc_score(y, prob[r], multi_class="ovr", average="macro")
    rows.append(s)
    print(f"{r}: macro_f1 {s['macro_f1']:.4f} (results.json said {REF[r]})")
    if abs(s["macro_f1"] - REF[r]) > 0.002: print("  WARNING: mismatch, blob may be corrupted")
pt = pd.DataFrame(rows).set_index("run"); pt.to_csv(OUT / "overall_metrics.csv")

# 2. per-class metrics and confusion matrices
pc = []
for r in runs:
    cm = cm_of(y, pred[r]); pr, rc, f1 = prf(cm)
    pd.DataFrame(cm, index=CL, columns=CL).to_csv(OUT / f"confusion_{r}.csv")
    for i, c in enumerate(CL):
        pc.append(dict(run=r, cls=c, support=int(cm[i].sum()), precision=pr[i], recall=rc[i], f1=f1[i]))
pd.DataFrame(pc).to_csv(OUT / "per_class_metrics.csv", index=False)

# 3. lesion-level paired bootstrap
B = 2000; rng = np.random.default_rng(0)
groups = list(base.groupby("lesion_id").indices.values())
mets = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
boot = {r: {m: np.empty(B) for m in mets} for r in runs}
for b in range(B):
    idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
    for r in runs:
        s = summ(y[idx], pred[r][idx])
        for m in mets: boot[r][m][b] = s[m]
ci = []
for r in runs:
    for m in mets:
        lo, hi = np.percentile(boot[r][m], [2.5, 97.5])
        ci.append(dict(run=r, metric=m, point=pt.loc[r, m], ci_low=lo, ci_high=hi))
pd.DataFrame(ci).to_csv(OUT / "bootstrap_ci.csv", index=False)

pairs = [("E3_fusion", "E1_effnet_b3"), ("E2_deit_small", "E1_effnet_b3"),
         ("E3_fusion", "E2_deit_small"), ("E4_effnet_b3_cw", "E1_effnet_b3")]
cmp_rows = []
for a, b_ in pairs:
    d = boot[a]["macro_f1"] - boot[b_]["macro_f1"]
    ca, cb = pred[a] == y, pred[b_] == y
    n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
    pv = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    cmp_rows.append(dict(model_a=a, model_b=b_, dF1=pt.loc[a, "macro_f1"] - pt.loc[b_, "macro_f1"],
                         dF1_ci_low=np.percentile(d, 2.5), dF1_ci_high=np.percentile(d, 97.5),
                         a_right_b_wrong=n10, a_wrong_b_right=n01, mcnemar_exact_p=pv))
pd.DataFrame(cmp_rows).to_csv(OUT / "paired_comparisons.csv", index=False)

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print("\nOVERALL\n", pt.round(4))
print("\nBOOTSTRAP 95% CI (macro_f1 and accuracy)\n",
      pd.DataFrame(ci).query("metric in ['macro_f1','accuracy']").round(4).to_string(index=False))
print("\nPAIRED COMPARISONS\n", pd.DataFrame(cmp_rows).round(4).to_string(index=False))
print("\nPER-CLASS RECALL / F1\n", pd.DataFrame(pc).pivot(index="cls", columns="run", values=["recall", "f1"]).round(3))
print("\nSaved to", OUT.resolve())
