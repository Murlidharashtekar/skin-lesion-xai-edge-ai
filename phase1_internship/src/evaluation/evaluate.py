from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from src.data.dataset import HAM10000Dataset
from src.models.efficientnet_b3 import create_model
from src.data.preprocessing import get_test_transforms


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_CSV = PROJECT_ROOT.parent / "data" / "splits" / "test.csv"

IMAGE_DIR = PROJECT_ROOT.parent / "data" / "raw" / "ham10000" / "images"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "efficientnet_b3_best.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
NUM_WORKERS = 0
NUM_CLASSES = 7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("EFFICIENTNET-B3 BASELINE EVALUATION")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Test CSV: {TEST_CSV}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Image directory: {IMAGE_DIR}")
    print()

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test CSV not found:\n{TEST_CSV}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found:\n{IMAGE_DIR}"
        )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print("Loading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    print("Checkpoint loaded.")

    # --------------------------------------------------------
    # Class mapping
    # --------------------------------------------------------

    class_to_idx = checkpoint.get("class_to_idx", None)

    if class_to_idx is None:
        print("WARNING: class_to_idx not found in checkpoint.")
        class_to_idx = {
            "akiec": 0,
            "bcc": 1,
            "bkl": 2,
            "df": 3,
            "mel": 4,
            "nv": 5,
            "vasc": 6,
        }

    idx_to_class = {
        int(index): class_name
        for class_name, index in class_to_idx.items()
    }

    print("\nClass mapping:")
    for index in sorted(idx_to_class):
        print(f"  {index}: {idx_to_class[index]}")

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    test_df = pd.read_csv(TEST_CSV)

    print(f"Test samples: {len(test_df)}")

    print("\nTest CSV columns:")
    print(test_df.columns.tolist())

    # --------------------------------------------------------
    # Create dataset
    # --------------------------------------------------------

    test_dataset = HAM10000Dataset(
        dataframe=test_df,
        image_dir=IMAGE_DIR,
        transform=get_test_transforms(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    print(f"Test batches: {len(test_loader)}")

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print("\nCreating EfficientNet-B3...")

    model = create_model(
        num_classes=NUM_CLASSES,
        pretrained=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    all_predictions = []
    all_labels = []

    print("\nStarting evaluation...")

    with torch.no_grad():

        for batch_index, batch in enumerate(test_loader):

            # Support datasets returning (image, label)
            images, labels = batch

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            if (
                batch_index == 0
                or (batch_index + 1) % 10 == 0
                or batch_index == len(test_loader) - 1
            ):
                print(
                    f"Processed batch "
                    f"{batch_index + 1}/{len(test_loader)}"
                )

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall   : {recall:.4f}")
    print(f"Macro F1       : {f1:.4f}")

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    class_indices = sorted(idx_to_class.keys())

    target_names = [
        idx_to_class[index]
        for index in class_indices
    ]

    report = classification_report(
        all_labels,
        all_predictions,
        labels=class_indices,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = (
        OUTPUT_DIR / "classification_report.csv"
    )

    report_df.to_csv(report_path)

    print("\nClass-wise results:")
    print(report_df)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=class_indices,
    )

    cm_df = pd.DataFrame(
        cm,
        index=target_names,
        columns=target_names,
    )

    cm_path = (
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    cm_df.to_csv(cm_path)

    # --------------------------------------------------------
    # Save metrics JSON
    # --------------------------------------------------------

    metrics = {
        "model": "EfficientNet-B3",
        "device": str(DEVICE),
        "test_samples": int(len(all_labels)),
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "class_to_idx": class_to_idx,
    }

    metrics_path = (
        OUTPUT_DIR / "baseline_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Print confusion matrix
    # --------------------------------------------------------

    print("\nConfusion Matrix:")
    print(cm_df)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print("\nSaved files:")

    print(
        f"1. {metrics_path}"
    )

    print(
        f"2. {report_path}"
    )

    print(
        f"3. {cm_path}"
    )


if __name__ == "__main__":
    main()