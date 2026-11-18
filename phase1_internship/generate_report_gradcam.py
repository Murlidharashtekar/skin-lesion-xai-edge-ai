
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from PIL import Image

from src.data.preprocessing import get_val_transforms
from src.models.efficientnet_b3 import create_model


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path("..").resolve()

TEST_CSV = PROJECT_ROOT / "data" / "splits" / "test.csv"

IMAGE_DIR = (
    PROJECT_ROOT / "data" / "raw" / "ham10000" / "images"
)

CHECKPOINT_PATH = (
    Path("outputs") / "checkpoints" / "efficientnet_b3_best.pth"
)

OUTPUT_DIR = Path("outputs") / "report_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 7

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# GRAD-CAM CLASS
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self.save_activation
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, inputs, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_index):

        self.model.zero_grad()

        output = self.model(input_tensor)
        score = output[0, class_index]
        score.backward()

        weights = self.gradients.mean(
            dim=(2, 3), keepdim=True
        )

        cam = (weights * self.activations).sum(
            dim=1, keepdim=True
        )

        cam = torch.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# FIND CORRECT AND INCORRECT PREDICTIONS
# ============================================================

def main():

    print("=" * 65)
    print("GENERATING REPORT GRAD-CAM FIGURES")
    print("=" * 65)

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    test_df = pd.read_csv(TEST_CSV)

    print(f"\nTest images: {len(test_df)}")
    print(f"Device: {DEVICE}")

    # Load model checkpoint
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    class_to_idx = checkpoint.get("class_to_idx")

    if class_to_idx is None:
        raise ValueError("class_to_idx is missing from checkpoint.")

    class_to_idx = {
        str(name): int(index)
        for name, index in class_to_idx.items()
    }

    idx_to_class = {
        index: name
        for name, index in class_to_idx.items()
    }

    model = create_model(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    transform = get_val_transforms()

    correct_case = None
    incorrect_case = None

    print("\nSearching test images...")

    # Evaluate test images
    for count, row in enumerate(test_df.itertuples(index=False), start=1):

        image_id = str(row.image_id)
        actual_class = str(row.dx).strip()

        image_path = IMAGE_DIR / f"{image_id}.jpg"

        if not image_path.exists():
            continue

        if actual_class not in class_to_idx:
            continue

        pil_image = Image.open(image_path).convert("RGB")
        input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)

            predicted_index = int(
                torch.argmax(probabilities, dim=1).item()
            )

            confidence = float(
                probabilities[0, predicted_index].item()
            )

        predicted_class = idx_to_class[predicted_index]

        case = {
            "image_id": image_id,
            "image_path": image_path,
            "actual": actual_class,
            "predicted": predicted_class,
            "predicted_index": predicted_index,
            "confidence": confidence
        }

        if predicted_class == actual_class and correct_case is None:
            correct_case = case
            print(
                f"Correct example found: {image_id} "
                f"({actual_class})"
            )

        elif predicted_class != actual_class and incorrect_case is None:
            incorrect_case = case
            print(
                f"Misclassified example found: {image_id} "
                f"(Actual: {actual_class}, "
                f"Predicted: {predicted_class})"
            )

        if correct_case is not None and incorrect_case is not None:
            break

        if count % 50 == 0:
            print(f"Checked {count} test images...")

    if correct_case is None:
        raise RuntimeError(
            "No correctly classified test image was found."
        )

    if incorrect_case is None:
        raise RuntimeError(
            "No misclassified test image was found."
        )

    # ========================================================
    # GENERATE GRAD-CAM FIGURES
    # ========================================================

    gradcam = GradCAM(
        model,
        model.conv_head
    )

    selected_cases = [
        (
            correct_case,
            "Figure 6.6 - Correctly Classified Example",
            "figure_6_6_gradcam_correct.png"
        ),
        (
            incorrect_case,
            "Figure 6.7 - Misclassified Example",
            "figure_6_7_gradcam_misclassified.png"
        )
    ]

    try:

        for case, figure_title, filename in selected_cases:

            pil_image = Image.open(
                case["image_path"]
            ).convert("RGB")

            input_tensor = transform(
                pil_image
            ).unsqueeze(0).to(DEVICE)

            cam = gradcam.generate(
                input_tensor,
                case["predicted_index"]
            )

            original_rgb = np.array(pil_image)

            height, width = original_rgb.shape[:2]

            cam_resized = cv2.resize(cam, (width, height))

            heatmap = cv2.applyColorMap(
                np.uint8(255 * cam_resized),
                cv2.COLORMAP_JET
            )

            heatmap = cv2.cvtColor(
                heatmap,
                cv2.COLOR_BGR2RGB
            )

            overlay = cv2.addWeighted(
                original_rgb,
                0.60,
                heatmap,
                0.40,
                0
            )

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            axes[0].imshow(original_rgb)
            axes[0].set_title("Original Skin Lesion Image")
            axes[0].axis("off")

            axes[1].imshow(overlay)
            axes[1].set_title("Grad-CAM Heatmap Overlay")
            axes[1].axis("off")

            result_text = (
                f"Actual: {case['actual']}  |  "
                f"Predicted: {case['predicted']}  |  "
                f"Confidence: {case['confidence'] * 100:.2f}%"
            )

            fig.suptitle(
                figure_title + "\n" + result_text,
                fontsize=12
            )

            plt.tight_layout()

            output_path = OUTPUT_DIR / filename

            plt.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

            print(f"\nSaved: {output_path}")
            print(f"Image ID: {case['image_id']}")
            print(f"Actual class: {case['actual']}")
            print(f"Predicted class: {case['predicted']}")
            print(
                f"Confidence: {case['confidence'] * 100:.2f}%"
            )

    finally:
        gradcam.close()

    print("\n" + "=" * 65)
    print("FIGURES 6.6 AND 6.7 GENERATED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()
