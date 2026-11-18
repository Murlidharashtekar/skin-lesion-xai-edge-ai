import torch.nn as nn
import timm


NUM_CLASSES = 7


def create_model(
    num_classes=NUM_CLASSES,
    pretrained=True
):
    """
    Create EfficientNet-B3 model.

    pretrained=True:
        Loads ImageNet pretrained weights.

    num_classes=7:
        HAM10000 seven-class classification.
    """

    model = timm.create_model(
        "efficientnet_b3",
        pretrained=pretrained,
        num_classes=num_classes
    )

    return model


if __name__ == "__main__":

    model = create_model()

    print(model)

    print("\nModel created successfully.")