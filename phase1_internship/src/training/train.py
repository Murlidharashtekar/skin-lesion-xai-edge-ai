from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from src.data.dataset import HAM10000Dataset
from src.data.preprocessing import (
    get_train_transforms,
    get_val_transforms
)

from src.models.efficientnet_b3 import create_model


# ============================================================
# Configuration
# ============================================================

SEED = 42

NUM_CLASSES = 7

EPOCHS = 2

BATCH_SIZE = 16

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

EARLY_STOPPING_PATIENCE = 5


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "train.csv"
)

VAL_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "validation.csv"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ham10000"
    / "images"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "phase1_internship"
    / "outputs"
    / "checkpoints"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "phase1_internship"
    / "outputs"
    / "metrics"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    targets,
    predictions
):

    accuracy = accuracy_score(
        targets,
        predictions
    )

    precision = precision_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1
    }


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    all_targets = []

    all_predictions = []

    total_batches = len(loader)

    print(f"Training batches: {total_batches}", flush=True)

    for batch_idx, (images, labels) in enumerate(loader, start=1):

        images = images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_targets.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        # Show progress every 10 batches
        if batch_idx % 10 == 0 or batch_idx == 1:

            percent = (
                batch_idx / total_batches
            ) * 100

            print(
                f"Batch {batch_idx}/{total_batches} "
                f"({percent:.1f}%) | "
                f"Loss: {loss.item():.4f}",
                flush=True
            )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_targets,
        all_predictions
    )

    return epoch_loss, metrics

# ============================================================
# Validation
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    all_targets = []

    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                torch.argmax(
                    outputs,
                    dim=1
                )
            )

            all_targets.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_targets,
        all_predictions
    )

    return epoch_loss, metrics


# ============================================================
# Main training
# ============================================================

def main():

    set_seed()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("EFFICIENTNET-B3 TRAINING")
    print("=" * 70)

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )


    # --------------------------------------------------------
    # Load metadata splits
    # --------------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_CSV
    )

    val_df = pd.read_csv(
        VAL_CSV
    )

    print(
        f"\nTraining images: {len(train_df)}"
    )

    print(
        f"Validation images: {len(val_df)}"
    )


    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_dataset = HAM10000Dataset(
        dataframe=train_df,
        image_dir=IMAGE_DIR,
        transform=get_train_transforms()
    )

    val_dataset = HAM10000Dataset(
        dataframe=val_df,
        image_dir=IMAGE_DIR,
        transform=get_val_transforms()
    )


    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------



    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print("\nTesting DataLoader...", flush=True)

    test_images, test_labels = next(iter(train_loader))

    print(
        f"First batch loaded successfully: "
        f"{test_images.shape}",
        flush=True
    )

    print(
        f"Labels shape: {test_labels.shape}",
        flush=True
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        num_classes=NUM_CLASSES,
        pretrained=True
    )

    model = model.to(device)


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )


    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = []

    best_val_f1 = -float("inf")

    epochs_without_improvement = 0


    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        train_loss, train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )
        )

        val_loss, val_metrics = (
            validate(
                model,
                val_loader,
                criterion,
                device
            )
        )


        scheduler.step(
            val_metrics["macro_f1"]
        )


        current_lr = (
            optimizer.param_groups[0]["lr"]
        )


        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_metrics['accuracy']:.4f}"
        )

        print(
            f"Train Macro F1: "
            f"{train_metrics['macro_f1']:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"Val Macro F1: "
            f"{val_metrics['macro_f1']:.4f}"
        )

        print(
            f"Learning Rate: {current_lr:.7f}"
        )


        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history.append({
            "epoch": epoch,

            "train_loss": train_loss,

            "train_accuracy":
                train_metrics["accuracy"],

            "train_macro_f1":
                train_metrics["macro_f1"],

            "val_loss": val_loss,

            "val_accuracy":
                val_metrics["accuracy"],

            "val_macro_f1":
                val_metrics["macro_f1"],

            "learning_rate": current_lr
        })


        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if (
            val_metrics["macro_f1"]
            > best_val_f1
        ):

            best_val_f1 = (
                val_metrics["macro_f1"]
            )

            epochs_without_improvement = 0

            checkpoint_path = (
                CHECKPOINT_DIR
                / "efficientnet_b3_best.pth"
            )

            torch.save({

                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "best_val_macro_f1":
                    best_val_f1,

                "class_to_idx":
                    train_dataset.class_to_idx,

                "seed":
                    SEED

            }, checkpoint_path)

            print(
                "\n*** Best model saved ***"
            )

            print(
                checkpoint_path
            )

        else:

            epochs_without_improvement += 1


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break


    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_file = (
        METRICS_DIR
        / "efficientnet_b3_training_history.json"
    )

    with open(
        history_file,
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )


    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nBest Validation Macro F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        f"\nTraining history saved to:"
    )

    print(history_file)


if __name__ == "__main__":
    main()