from torchvision import transforms


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 300

# ImageNet statistics
IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# ============================================================
# Training Transform
# ============================================================

def get_train_transforms():
    """
    Transformations applied only to training images.

    Includes:
    - Resize
    - Horizontal flip
    - Vertical flip
    - Small rotation
    - Color variation
    - Normalization
    """

    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomVerticalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=15
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


# ============================================================
# Validation Transform
# ============================================================

def get_val_transforms():
    """
    Transformations applied to validation images.

    No random augmentation is used.
    """

    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


# ============================================================
# Test Transform
# ============================================================

def get_test_transforms():
    """
    Transformations applied to test images.

    No random augmentation is used.
    """

    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    test_transform = get_test_transforms()

    print("Training transformations:")
    print(train_transform)

    print("\nValidation transformations:")
    print(val_transform)

    print("\nTest transformations:")
    print(test_transform)

    print(
        f"\nImage size: {IMAGE_SIZE} x {IMAGE_SIZE}"
    )