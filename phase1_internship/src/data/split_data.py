from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ham10000"
    / "metadata"
    / "HAM10000_metadata.csv"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
)

SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(METADATA_FILE)

required_columns = {"image_id", "dx", "lesion_id"}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

np.random.seed(SEED)


# --------------------------------------------------
# First split:
# 85% development + 15% test
#
# Group = lesion_id
# Stratification = dx
# --------------------------------------------------

sgkf_test = StratifiedGroupKFold(
    n_splits=20,
    shuffle=True,
    random_state=SEED
)

train_val_idx, test_idx = next(
    sgkf_test.split(
        df,
        y=df["dx"],
        groups=df["lesion_id"]
    )
)

train_val = df.iloc[train_val_idx].copy()
test = df.iloc[test_idx].copy()


# --------------------------------------------------
# Second split:
# 70% train + 15% validation
#
# Within the 85% development set:
#
# 70 / 85 ≈ 82.35%
# 15 / 85 ≈ 17.65%
# --------------------------------------------------

val_fraction = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)

n_splits_val = 20

sgkf_val = StratifiedGroupKFold(
    n_splits=n_splits_val,
    shuffle=True,
    random_state=SEED
)

train_idx, val_idx = next(
    sgkf_val.split(
        train_val,
        y=train_val["dx"],
        groups=train_val["lesion_id"]
    )
)

train = train_val.iloc[train_idx].copy()
val = train_val.iloc[val_idx].copy()


# --------------------------------------------------
# Save splits
# --------------------------------------------------

train_file = SPLIT_DIR / "train.csv"
val_file = SPLIT_DIR / "validation.csv"
test_file = SPLIT_DIR / "test.csv"

train.to_csv(train_file, index=False)
val.to_csv(val_file, index=False)
test.to_csv(test_file, index=False)


# --------------------------------------------------
# Verification
# --------------------------------------------------

print("=" * 70)
print("HAM10000 SPLIT RESULTS")
print("=" * 70)

print(f"\nSeed: {SEED}")

print("\nNumber of images:")
print(f"Train      : {len(train)}")
print(f"Validation : {len(val)}")
print(f"Test       : {len(test)}")

print("\nPercentages:")

total = len(df)

print(f"Train      : {len(train) / total * 100:.2f}%")
print(f"Validation : {len(val) / total * 100:.2f}%")
print(f"Test       : {len(test) / total * 100:.2f}%")


# --------------------------------------------------
# Class distributions
# --------------------------------------------------

print("\nTRAIN CLASS DISTRIBUTION")
print(train["dx"].value_counts())

print("\nVALIDATION CLASS DISTRIBUTION")
print(val["dx"].value_counts())

print("\nTEST CLASS DISTRIBUTION")
print(test["dx"].value_counts())


# --------------------------------------------------
# Leakage checks
# --------------------------------------------------

train_lesions = set(train["lesion_id"])
val_lesions = set(val["lesion_id"])
test_lesions = set(test["lesion_id"])


train_val_overlap = train_lesions & val_lesions
train_test_overlap = train_lesions & test_lesions
val_test_overlap = val_lesions & test_lesions


print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

print(
    "Train ∩ Validation:",
    len(train_val_overlap)
)

print(
    "Train ∩ Test:",
    len(train_test_overlap)
)

print(
    "Validation ∩ Test:",
    len(val_test_overlap)
)


if (
    len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(val_test_overlap) == 0
):
    print("\nPASS: No lesion_id overlap between splits.")
else:
    raise RuntimeError(
        "DATA LEAKAGE DETECTED!"
    )


# --------------------------------------------------
# Image ID leakage check
# --------------------------------------------------

train_images = set(train["image_id"])
val_images = set(val["image_id"])
test_images = set(test["image_id"])

assert not (train_images & val_images)
assert not (train_images & test_images)
assert not (val_images & test_images)

print("PASS: No image_id overlap between splits.")


# --------------------------------------------------
# Final status
# --------------------------------------------------

print("\n" + "=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)

print(f"\nSaved:")
print(train_file)
print(val_file)
print(test_file)