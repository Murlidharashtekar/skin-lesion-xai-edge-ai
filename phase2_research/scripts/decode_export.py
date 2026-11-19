import sys, re, json, zlib, base64
from pathlib import Path
import numpy as np, pandas as pd

CL = ["akiec","bcc","bkl","df","mel","nv","vasc"]
def f1_macro(y, p):
    cm = np.bincount(y*7+p, minlength=49).reshape(7,7)
    tp = np.diag(cm).astype(float); d = cm.sum(0)+cm.sum(1)
    return float(np.divide(2*tp, d, out=np.zeros(7), where=d>0).mean())

for f in sys.argv[1:]:
    t = Path(f).read_text(encoding="utf-8", errors="ignore").splitlines()
    head = next(l for l in t if l.startswith("#TAG"))
    m = re.match(r"#TAG (\S+) #LEN (\d+) #CRC (\d+)", head)
    tag, n, crc = m.group(1), int(m.group(2)), int(m.group(3))
    blob = "".join(re.sub(r"[^A-Za-z0-9+/=]", "", l) for l in t if not l.startswith("#") and l.strip())
    print(tag, "length", len(blob), "expected", n, "| crc ok:", zlib.crc32(blob.encode()) == crc)
    assert len(blob) == n and zlib.crc32(blob.encode()) == crc, "corrupted copy"
    o = json.loads(zlib.decompress(base64.b64decode(blob)))
    out = Path("phase2_research/outputs/seeds")
    for s in ("val", "test"):
        y = np.array(o[s]["y"]); p = np.array(o[s]["p"]) / 10000
        pd.DataFrame({"y_true": y, **{f"p_{c}": p[:, i] for i, c in enumerate(CL)}}).to_csv(out / f"{tag}_{s}.csv", index=False)
        print(f"  {s}: n={len(y)} macro_f1={f1_macro(y, p.argmax(1)):.4f}")
    print("  results.json test macro_f1:", round(o["results"]["test"]["macro_f1"], 4))
