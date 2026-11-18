
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
PROJECT_ROOT = Path.cwd().parent
OUTPUT_DIR = Path("outputs/report_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# ---------------------------------------
# FIGURE 6.1: Dataset class distribution
# ---------------------------------------

metadata_path = (
    PROJECT_ROOT / "data/raw/ham10000/metadata/"
    "HAM10000_metadata.csv"
)

df = pd.read_csv(metadata_path)
counts = df["dx"].value_counts().reindex(CLASSES, fill_value=0)

plt.figure(figsize=(10, 6))
bars = plt.bar(counts.index, counts.values)

plt.title("HAM10000 Dataset Class Distribution")
plt.xlabel("Skin Disease Classes")
plt.ylabel("Number of Images")

for bar, count in zip(bars, counts.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(count),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "figure_6_1_class_distribution.png",
    dpi=300
)
plt.close()

# ---------------------------------------
# LOAD TRAINING HISTORY
# ---------------------------------------

history_path = Path(
    "outputs/metrics/efficientnet_b3_training_history.json"
)

with open(history_path, "r") as f:
    history = json.load(f)

if isinstance(history, dict):
    history = history.get("history", history.get("epochs", []))

if not history:
    raise ValueError("Training history is empty.")

epochs = [x["epoch"] for x in history]

train_accuracy = [x["train_accuracy"] for x in history]
val_accuracy = [x["val_accuracy"] for x in history]

train_loss = [x["train_loss"] for x in history]
val_loss = [x["val_loss"] for x in history]

# ---------------------------------------
# FIGURE 6.2: ACCURACY
# ---------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    train_accuracy,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_accuracy,
    marker="o",
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_6_2_accuracy.png",
    dpi=300
)
plt.close()

# ---------------------------------------
# FIGURE 6.3: LOSS
# ---------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    train_loss,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    marker="o",
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_6_3_loss.png",
    dpi=300
)
plt.close()

print("\nSUCCESS: FIGURES 6.1, 6.2 AND 6.3 GENERATED")

for file in sorted(OUTPUT_DIR.glob("figure_6_*.png")):
    print(file)
