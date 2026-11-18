from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
ROOT = Path.cwd()
META = ROOT / "data/raw/ham10000/metadata/HAM10000_metadata.csv"
OUT = ROOT / "data/splits_v2"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(META)

dev_idx, test_idx = next(StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED).split(df, df["dx"], df["lesion_id"]))
dev, test = df.iloc[dev_idx], df.iloc[test_idx]
tr_idx, va_idx = next(StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED).split(dev, dev["dx"], dev["lesion_id"]))
train, val = dev.iloc[tr_idx], dev.iloc[va_idx]

for name, d in (("train", train), ("validation", val), ("test", test)):
    d.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name}: {len(d)} ({100*len(d)/len(df):.1f}%)", d["dx"].value_counts().to_dict())
for key in ("lesion_id", "image_id"):
    for a, b, n in ((train, val, "train/val"), (train, test, "train/test"), (val, test, "val/test")):
        k = len(set(a[key]) & set(b[key]))
        print(f"{key} overlap {n}: {k}")
        assert k == 0, "LEAKAGE"
print("PASS. Saved to", OUT)
