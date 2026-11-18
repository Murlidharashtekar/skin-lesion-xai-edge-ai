import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
SPLITS = ROOT / "data" / "splits"
META = ROOT / "data" / "raw" / "ham10000" / "metadata" / "HAM10000_metadata.csv"
IMAGES = ROOT / "data" / "raw" / "ham10000" / "images"

names = ["train", "validation", "test"]
dfs = {n: pd.read_csv(SPLITS / f"{n}.csv") for n in names}
meta = pd.read_csv(META)
problems = []

print(f"Metadata: {len(meta)} images, {meta['lesion_id'].nunique()} lesions")
for n, d in dfs.items():
    share = 100 * len(d) / len(meta)
    print(f"\n[{n}] {len(d)} images ({share:.1f}%), {d['lesion_id'].nunique()} lesions")
    print("  classes:", d["dx"].value_counts().to_dict())

print()
for key in ("lesion_id", "image_id"):
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        k = len(set(dfs[a][key]) & set(dfs[b][key]))
        print(f"{key} overlap {a} vs {b}: {k}")
        if k:
            problems.append(f"{key} overlap {a}/{b}: {k}")

all_ids = pd.concat([d["image_id"] for d in dfs.values()])
if all_ids.duplicated().any():
    problems.append("duplicate image_id within or across splits")
missing = set(meta["image_id"]) - set(all_ids)
if missing:
    problems.append(f"{len(missing)} metadata images are in no split")

if IMAGES.exists():
    absent = [i for i in all_ids if not (IMAGES / f"{i}.jpg").exists()]
    print(f"\nSplit images missing on disk: {len(absent)}")
    if absent:
        problems.append(f"{len(absent)} split images not found on disk")
else:
    print("\nImage folder not found:", IMAGES)

print("\nRESULT:", "PASS" if not problems else "FAIL")
for p in problems:
    print(" -", p)
sys.exit(1 if problems else 0)
