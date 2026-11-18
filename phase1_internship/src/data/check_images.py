from pathlib import Path
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "ham10000" / "images"


def main():
    image_files = list(IMAGE_DIR.glob("*.jpg"))

    print(f"Checking {len(image_files)} images...\n")

    valid = 0
    corrupted = []

    for image_path in image_files:
        try:
            with Image.open(image_path) as img:
                img.verify()

            valid += 1

        except Exception as e:
            corrupted.append((image_path.name, str(e)))

    print(f"Valid images: {valid}")
    print(f"Corrupted images: {len(corrupted)}")

    if corrupted:
        print("\nCorrupted files:")

        for filename, error in corrupted:
            print(f"{filename}: {error}")

    else:
        print("\nPASS: All images can be opened successfully.")


if __name__ == "__main__":
    main()