
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
OUTPUT_DIR = Path("outputs/report_figures")
BASELINE_DIR = Path("outputs/baseline")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# ==========================================
# FIGURE 6.4 - CONFUSION MATRIX
# ==========================================

cm_path = BASELINE_DIR / "confusion_matrix.csv"

# First column contains actual class labels
cm_df = pd.read_csv(cm_path, index_col=0)

# Arrange rows and columns in the correct order
cm_df.index = cm_df.index.astype(str).str.strip()
cm_df.columns = cm_df.columns.astype(str).str.strip()
cm_df = cm_df.reindex(index=CLASSES, columns=CLASSES)

if cm_df.isnull().any().any():
    raise ValueError("Confusion matrix is missing one or more classes.")

cm = cm_df.to_numpy()

plt.figure(figsize=(9, 7))
plt.imshow(cm, interpolation="nearest", cmap="Blues")

plt.title("Confusion Matrix - EfficientNet-B3")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.xticks(range(7), CLASSES, rotation=45)
plt.yticks(range(7), CLASSES)

threshold = cm.max() / 2 if cm.max() > 0 else 0

for i in range(7):
    for j in range(7):
        plt.text(
            j, i, str(int(cm[i, j])),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black"
        )

plt.colorbar(label="Number of Images")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_6_4_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# FIGURE 6.5 - CLASSIFICATION REPORT
# ==========================================

report_path = BASELINE_DIR / "classification_report.csv"

# First column contains class labels
report = pd.read_csv(report_path, index_col=0)

report.index = report.index.astype(str).str.strip()

# Select the seven disease classes only
report = report.loc[report.index.isin(CLASSES)]
report = report.reindex(CLASSES)

if report.isnull().any().any():
    raise ValueError("One or more class rows are missing.")

columns = ["precision", "recall", "f1-score", "support"]
report = report[columns]

fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("off")

table = ax.table(
    cellText=report.round(3).values,
    rowLabels=report.index,
    colLabels=columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

plt.title(
    "Class-wise Classification Report - EfficientNet-B3",
    pad=25
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_6_5_classification_report.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# COMPLETION
# ==========================================

print("\nSUCCESS: FIGURES 6.4 AND 6.5 GENERATED")
print(OUTPUT_DIR / "figure_6_4_confusion_matrix.png")
print(OUTPUT_DIR / "figure_6_5_classification_report.png")
