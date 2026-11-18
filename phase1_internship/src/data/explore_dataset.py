from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Project paths
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "phase1_internship"
    / "outputs"
    / "figures"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load metadata
# --------------------------------------------------

df = pd.read_csv(METADATA_FILE)

print("=" * 70)
print("HAM10000 DATASET EXPLORATION")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# --------------------------------------------------
# Class distribution
# --------------------------------------------------

class_counts = df["dx"].value_counts()

print("\nClass distribution:")
print(class_counts)

print("\nClass percentages:")
print(
    (df["dx"].value_counts(normalize=True) * 100)
    .round(2)
)


# --------------------------------------------------
# Unique lesions
# --------------------------------------------------

print("\nUnique lesion IDs:")
print(df["lesion_id"].nunique())

print("\nImages per lesion:")
print(df.groupby("lesion_id").size().describe())


# --------------------------------------------------
# Images per class
# --------------------------------------------------

class_lesion_counts = (
    df.groupby("dx")["lesion_id"]
    .nunique()
    .sort_values(ascending=False)
)

print("\nUnique lesions per class:")
print(class_lesion_counts)


# --------------------------------------------------
# Missing values
# --------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())


# --------------------------------------------------
# Class distribution plot
# --------------------------------------------------

plt.figure(figsize=(10, 6))

class_counts.plot(kind="bar")

plt.title("HAM10000 Class Distribution")
plt.xlabel("Lesion Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=0)

plt.tight_layout()

output_file = OUTPUT_DIR / "class_distribution.png"
plt.savefig(output_file, dpi=300)
plt.close()

print(f"\nSaved plot to:")
print(output_file)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("EXPLORATION COMPLETE")
print("=" * 70)