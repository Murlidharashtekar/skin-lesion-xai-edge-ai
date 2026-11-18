from pathlib import Path
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "ham10000" / "images"
METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ham10000"
    / "metadata"
    / "HAM10000_metadata.csv"
)


def main():
    print("=" * 60)
    print("HAM10000 DATASET VERIFICATION")
    print("=" * 60)

    # Check metadata
    if not METADATA_FILE.exists():
        print(f"ERROR: Metadata not found:\n{METADATA_FILE}")
        return

    df = pd.read_csv(METADATA_FILE)

    print(f"\nMetadata rows: {len(df)}")
    print(f"Metadata columns: {list(df.columns)}")

    # Check image directory
    if not IMAGE_DIR.exists():
        print(f"\nERROR: Image directory not found:\n{IMAGE_DIR}")
        return

    image_files = list(IMAGE_DIR.glob("*.jpg"))

    print(f"\nImage files found: {len(image_files)}")

    # Create image lookup
    image_ids = {
        image_file.stem
        for image_file in image_files
    }

    # Check metadata image IDs
    metadata_ids = set(df["image_id"].astype(str))

    missing_images = metadata_ids - image_ids
    extra_images = image_ids - metadata_ids

    print(f"\nMetadata image IDs: {len(metadata_ids)}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Extra images: {len(extra_images)}")

    # Class distribution
    print("\nClass distribution:")
    print(df["dx"].value_counts())

    # Missing metadata values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Final status
    print("\n" + "=" * 60)

    if len(missing_images) == 0:
        print("PASS: Every metadata image_id has a corresponding image.")
    else:
        print("WARNING: Some metadata image IDs have no image file.")

    if len(extra_images) == 0:
        print("PASS: No extra image files were found.")
    else:
        print("WARNING: Some image files are not present in metadata.")

    print("=" * 60)


if __name__ == "__main__":
    main()