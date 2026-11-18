
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from PIL import Image

from src.data.preprocessing import get_val_transforms
from src.models.efficientnet_b3 import create_model


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 7

IMAGE_SIZE = 300


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "test.csv"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ham10000"
    / "images"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "phase1_internship"
    / "outputs"
    / "checkpoints"
    / "efficientnet_b3_best.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "phase1_internship"
    / "outputs"
    / "gradcam"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Grad-CAM
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model

        self.target_layer = target_layer

        self.activations = None

        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )

    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    def generate(
        self,
        input_tensor,
        class_index
    ):

        self.model.zero_grad()

        output = self.model(
            input_tensor
        )

        score = output[
            0,
            class_index
        ]

        score.backward()

        activations = self.activations

        gradients = self.gradients

        # Global average pooling of gradients

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted feature maps

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        # ReLU

        cam = torch.relu(cam)

        # Remove batch dimension

        cam = cam.squeeze(
            0
        ).squeeze(
            0
        )

        cam = cam.detach().cpu().numpy()

        # Normalize

        cam -= cam.min()

        if cam.max() > 0:

            cam /= cam.max()

        return cam

    def close(self):

        self.forward_handle.remove()

        self.backward_handle.remove()


# ============================================================
# Image overlay
# ============================================================

def create_overlay(
    original_image,
    cam
):

    height, width = (
        original_image.shape[:2]
    )

    cam_resized = cv2.resize(
        cam,
        (width, height)
    )

    heatmap = np.uint8(
        255 * cam_resized
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        original_image,
        0.60,
        heatmap,
        0.40,
        0
    )

    return overlay


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("EFFICIENTNET-B3 GRAD-CAM")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not TEST_CSV.exists():

        raise FileNotFoundError(
            f"Test CSV not found:\n{TEST_CSV}"
        )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    test_df = pd.read_csv(
        TEST_CSV
    )

    print(
        f"Test images available: "
        f"{len(test_df)}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = create_model(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    # --------------------------------------------------------
    # Class mapping
    # --------------------------------------------------------

    class_to_idx = checkpoint.get(
        "class_to_idx"
    )

    if class_to_idx is None:

        raise ValueError(
            "class_to_idx not found in checkpoint."
        )

    idx_to_class = {
        int(index): class_name
        for class_name, index
        in class_to_idx.items()
    }

    print(
        "\nClass mapping:"
    )

    for index in range(NUM_CLASSES):

        print(
            f"{index}: "
            f"{idx_to_class[index]}"
        )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    transform = get_val_transforms()

    # --------------------------------------------------------
    # Target layer
    # --------------------------------------------------------

    target_layer = model.conv_head

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer
    )

    # --------------------------------------------------------
    # Select images
    # --------------------------------------------------------

    selected_images = test_df.head(5)

    print(
        f"\nGenerating Grad-CAM for "
        f"{len(selected_images)} images..."
    )

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for index, row in selected_images.iterrows():

        image_name = str(
            row["image_id"]
        )

        image_path = (
            IMAGE_DIR
            / f"{image_name}.jpg"
        )

        if not image_path.exists():

            print(
                f"Skipping missing image: "
                f"{image_path}"
            )

            continue

        # Original image

        original = cv2.imread(
            str(image_path)
        )

        if original is None:

            print(
                f"Could not read: "
                f"{image_path}"
            )

            continue

        original_rgb = cv2.cvtColor(
            original,
            cv2.COLOR_BGR2RGB
        )

        # PIL image

        pil_image = Image.open(
            image_path
        ).convert("RGB")

        # Apply preprocessing

        input_tensor = transform(
            pil_image
        )

        input_tensor = (
            input_tensor
            .unsqueeze(0)
            .to(device)
        )

        # ----------------------------------------------------
        # Forward prediction
        # ----------------------------------------------------

        model.zero_grad()

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

        class_name = idx_to_class[
            predicted_class
        ]

        # ----------------------------------------------------
        # Generate Grad-CAM
        # ----------------------------------------------------

        cam = gradcam.generate(
            input_tensor,
            predicted_class
        )

        # ----------------------------------------------------
        # Create overlay
        # ----------------------------------------------------

        overlay = create_overlay(
            original,
            cam
        )

        # ----------------------------------------------------
        # Add prediction text
        # ----------------------------------------------------

        text = (
            f"Prediction: {class_name} "
            f"({confidence * 100:.2f}%)"
        )

        cv2.putText(
            overlay,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Save output
        # ----------------------------------------------------

        output_path = (
            OUTPUT_DIR
            / f"{image_name}_gradcam.jpg"
        )

        cv2.imwrite(
            str(output_path),
            overlay
        )

        print(
            f"\nImage: {image_name}"
        )

        print(
            f"Prediction: {class_name}"
        )

        print(
            f"Confidence: "
            f"{confidence * 100:.2f}%"
        )

        print(
            f"Saved: {output_path}"
        )

    gradcam.close()

    print(
        "\n" + "=" * 70
    )

    print(
        "GRAD-CAM COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
