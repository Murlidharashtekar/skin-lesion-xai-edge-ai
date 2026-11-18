from pathlib import Path
import base64
import io

import cv2
import numpy as np
import torch

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

from src.data.preprocessing import get_val_transforms
from src.models.efficientnet_b3 import create_model

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 7

BATCH_SIZE = 1


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "efficientnet_b3_best.pth"
)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Skin Lesion XAI API",
    description=(
        "EfficientNet-B3 skin lesion classification "
        "with Grad-CAM explanation."
    ),
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Load model ONCE
# ============================================================

print("=" * 70)
print("LOADING EFFICIENTNET-B3 MODEL")
print("=" * 70)

if not CHECKPOINT_PATH.exists():

    raise FileNotFoundError(
        f"Checkpoint not found:\n"
        f"{CHECKPOINT_PATH}"
    )


model = create_model(
    num_classes=NUM_CLASSES,
    pretrained=False
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


# ============================================================
# Class mapping
# ============================================================

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
    f"Device: {DEVICE}"
)

print(
    f"Model loaded from:\n{CHECKPOINT_PATH}"
)

print(
    "\nClass mapping:"
)

for index in range(NUM_CLASSES):

    print(
        f"{index}: {idx_to_class[index]}"
    )

print(
    "\nModel loaded successfully."
)


# ============================================================
# Preprocessing
# ============================================================

transform = get_val_transforms()


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

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = torch.relu(
            cam
        )

        cam = cam.squeeze(
            0
        ).squeeze(
            0
        )

        cam = (
            cam.detach()
            .cpu()
            .numpy()
        )

        cam -= cam.min()

        if cam.max() > 0:

            cam /= cam.max()

        return cam


    def close(self):

        self.forward_handle.remove()

        self.backward_handle.remove()


# ============================================================
# Initialize Grad-CAM ONCE
# ============================================================

target_layer = model.conv_head

gradcam = GradCAM(
    model=model,
    target_layer=target_layer
)


# ============================================================
# Health endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Skin Lesion XAI API is running",
        "model": "EfficientNet-B3",
        "device": str(DEVICE),
        "classes": NUM_CLASSES
    }


# ============================================================
# Health check
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "model": "EfficientNet-B3"
    }


# ============================================================
# Create Grad-CAM overlay
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
# Convert image to Base64
# ============================================================

def image_to_base64(
    image
):

    success, encoded_image = cv2.imencode(
        ".jpg",
        image
    )

    if not success:

        raise ValueError(
            "Could not encode explanation image."
        )

    image_bytes = (
        encoded_image
        .tobytes()
    )

    encoded_string = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return encoded_string


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/jpg",
        "application/octet-stream",
        "image/webp",
    }

    if file.content_type and file.content_type.lower() not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image content type: {file.content_type}"
        )

    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )



    try:

        pil_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )


    # --------------------------------------------------------
    # Original OpenCV image
    # --------------------------------------------------------

    original_rgb = np.array(
        pil_image
    )

    original_bgr = cv2.cvtColor(
        original_rgb,
        cv2.COLOR_RGB2BGR
    )


    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    input_tensor = transform(
        pil_image
    )

    input_tensor = (
        input_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

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


    # --------------------------------------------------------
    # Class name
    # --------------------------------------------------------

    predicted_label = idx_to_class[
        predicted_class
    ]


    # --------------------------------------------------------
    # Generate Grad-CAM
    # --------------------------------------------------------

    cam = gradcam.generate(
        input_tensor,
        predicted_class
    )


    # --------------------------------------------------------
    # Create explanation image
    # --------------------------------------------------------

    overlay = create_overlay(
        original_bgr,
        cam
    )


    # --------------------------------------------------------
    # Convert explanation image
    # --------------------------------------------------------

    explanation_base64 = image_to_base64(
        overlay
    )


    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return {

        "prediction": predicted_label,

        "class_index": predicted_class,

        "confidence": round(
            confidence,
            4
        ),

        "model": "EfficientNet-B3",

        "explanation": "Grad-CAM",

        "explanation_image_base64":
            explanation_base64

    }
