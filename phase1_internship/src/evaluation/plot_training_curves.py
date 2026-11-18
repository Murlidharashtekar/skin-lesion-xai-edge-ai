from pathlib import Path
import json

import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "efficientnet_b3_training_history.json"
)

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("TRAINING CURVE GENERATION")
    print("=" * 60)

    print(f"History file: {HISTORY_PATH}")
    print(f"Output directory: {FIGURES_DIR}")

    # --------------------------------------------------------
    # Check history file
    # --------------------------------------------------------

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Training history not found:\n{HISTORY_PATH}"
        )

    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------

    with open(HISTORY_PATH, "r", encoding="utf-8") as file:
        history = json.load(file)

    if not isinstance(history, list):
        raise ValueError(
            "Expected training history to be a list of epoch records."
        )

    if len(history) == 0:
        raise ValueError(
            "Training history is empty."
        )

    print(f"\nNumber of recorded epochs: {len(history)}")

    # --------------------------------------------------------
    # Extract values
    # --------------------------------------------------------

    epochs = [
        record["epoch"]
        for record in history
    ]

    train_loss = [
        record["train_loss"]
        for record in history
    ]

    val_loss = [
        record["val_loss"]
        for record in history
    ]

    train_accuracy = [
        record["train_accuracy"]
        for record in history
    ]

    val_accuracy = [
        record["val_accuracy"]
        for record in history
    ]

    train_f1 = [
        record["train_macro_f1"]
        for record in history
    ]

    val_f1 = [
        record["val_macro_f1"]
        for record in history
    ]

    learning_rates = [
        record["learning_rate"]
        for record in history
    ]

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\nTraining summary:")
    print(f"First epoch : {epochs[0]}")
    print(f"Last epoch  : {epochs[-1]}")

    print(
        f"Best validation accuracy: "
        f"{max(val_accuracy):.4f}"
    )

    print(
        f"Best validation Macro F1: "
        f"{max(val_f1):.4f}"
    )

    print(
        f"Best training accuracy: "
        f"{max(train_accuracy):.4f}"
    )

    # ========================================================
    # 1. LOSS CURVE
    # ========================================================

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

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "EfficientNet-B3 Training and Validation Loss"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_path = FIGURES_DIR / "loss_curve.png"

    plt.savefig(
        loss_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nSaved: {loss_path}")

    # ========================================================
    # 2. ACCURACY CURVE
    # ========================================================

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

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "EfficientNet-B3 Training and Validation Accuracy"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    accuracy_path = FIGURES_DIR / "accuracy_curve.png"

    plt.savefig(
        accuracy_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {accuracy_path}")

    # ========================================================
    # 3. MACRO F1 CURVE
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        train_f1,
        marker="o",
        label="Training Macro F1"
    )

    plt.plot(
        epochs,
        val_f1,
        marker="o",
        label="Validation Macro F1"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title(
        "EfficientNet-B3 Training and Validation Macro F1"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    f1_path = FIGURES_DIR / "macro_f1_curve.png"

    plt.savefig(
        f1_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {f1_path}")

    # ========================================================
    # 4. LEARNING RATE CURVE
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        learning_rates,
        marker="o",
        label="Learning Rate"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title(
        "EfficientNet-B3 Learning Rate Schedule"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    lr_path = FIGURES_DIR / "learning_rate_curve.png"

    plt.savefig(
        lr_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {lr_path}")

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 60)
    print("TRAINING CURVES COMPLETE")
    print("=" * 60)

    print("\nGenerated files:")

    print(f"1. {loss_path}")
    print(f"2. {accuracy_path}")
    print(f"3. {f1_path}")
    print(f"4. {lr_path}")


if __name__ == "__main__":
    main()