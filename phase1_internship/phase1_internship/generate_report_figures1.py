
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs/report_figures", exist_ok=True)

# Figure 6.1: Dataset class distribution
metadata_path = "data/raw/ham10000/metadata/HAM10000_metadata.csv"

df = pd.read_csv(metadata_path)
class_counts = df["dx"].value_counts().reindex(
    ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
    fill_value=0
)

plt.figure(figsize=(9, 5))
class_counts.plot(kind="bar")
plt.title("HAM10000 Dataset Class Distribution")
plt.xlabel("Skin Disease Classes")
plt.ylabel("Number of Images")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/report_figures/figure_6_1_class_distribution.png", dpi=300)
plt.close()

# Load training history
history_path = "outputs/metrics/efficientnet_b3_training_history.json"

with open(history_path, "r") as f:
    history = json.load(f)

epochs = range(1, len(history) + 1)

# Figure 6.2: Accuracy
plt.figure(figsize=(9, 5))
plt.plot(epochs, [x["train_acc"] for x in history], label="Training Accuracy")
plt.plot(epochs, [x["val_acc"] for x in history], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/report_figures/figure_6_2_accuracy.png", dpi=300)
plt.close()

# Figure 6.3: Loss
plt.figure(figsize=(9, 5))
plt.plot(epochs, [x["train_loss"] for x in history], label="Training Loss")
plt.plot(epochs, [x["val_loss"] for x in history], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/report_figures/figure_6_3_loss.png", dpi=300)
plt.close()

print("Figures 6.1, 6.2 and 6.3 generated successfully.")
