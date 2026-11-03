
import hashlib
import shutil
from pathlib import Path

import pandas as pd


# ==================================================
# 1. Project configuration
# ==================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    ROOT_DIR / "reports" / "classification_split_manifest.csv"
)

SOURCE_DIR = ROOT_DIR / "datasets" / "classification"

OUTPUT_DIR = ROOT_DIR / "datasets" / "classification_clean"

SPLIT_MAPPING = {
    "Train": "Train",
    "Validation": "Val",
    "Test": "Test",
}


# ==================================================
# 2. Calculate SHA-256
# ==================================================

def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# ==================================================
# 3. Resolve source image path
# ==================================================

def resolve_source_path(path_value):
    source_path = Path(path_value)

    # Use the original path if it exists.
    if source_path.is_file():
        return source_path

    # Support Docker paths when the manifest contains
    # absolute paths from the host machine.
    normalized_path = source_path.as_posix()
    marker = "/datasets/classification/"

    if marker in normalized_path:
        relative_path = normalized_path.split(marker, 1)[1]
        docker_path = SOURCE_DIR / relative_path

        if docker_path.is_file():
            return docker_path

    raise FileNotFoundError(
        f"Source image not found: {path_value}"
    )


# ==================================================
# 4. Prepare classification dataset
# ==================================================

def prepare_classification_dataset(overwrite=False):

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    if OUTPUT_DIR.exists():

        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {OUTPUT_DIR}\n"
                "Remove it manually or use overwrite=True."
            )

        shutil.rmtree(OUTPUT_DIR)

    print("Loading classification manifest...")

    df = pd.read_csv(MANIFEST_PATH)

    required_columns = {
        "dataset_split",
        "class",
        "filename",
        "sha256",
        "path",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing manifest columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Classification manifest is empty.")

    if df.duplicated(subset=["path"]).any():
        raise ValueError(
            "Duplicate source paths found in the manifest."
        )

    invalid_splits = set(df["dataset_split"]) - set(SPLIT_MAPPING)

    if invalid_splits:
        raise ValueError(
            f"Unexpected dataset splits: {invalid_splits}"
        )

    class_names = sorted(df["class"].unique().tolist())

    print(f"Classes: {class_names}")
    print(f"Images to process: {len(df)}")

    # Create clean dataset directories.
    for split in SPLIT_MAPPING.values():
        for class_name in class_names:
            (
                OUTPUT_DIR
                / split
                / class_name
            ).mkdir(parents=True, exist_ok=True)

    copied_count = 0

    # Copy and verify every image.
    for index, row in df.iterrows():

        source_path = resolve_source_path(row["path"])

        expected_hash = str(row["sha256"]).strip().lower()
        actual_hash = calculate_sha256(source_path)

        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch: {source_path}"
            )

        split = SPLIT_MAPPING[row["dataset_split"]]
        class_name = str(row["class"])

        if source_path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png"
        }:
            raise ValueError(
                f"Unsupported image format: {source_path}"
            )

        destination_dir = OUTPUT_DIR / split / class_name
        destination_path = destination_dir / source_path.name

        if destination_path.exists():
            raise FileExistsError(
                f"Duplicate destination filename: {destination_path}"
            )

        shutil.copy2(source_path, destination_path)

        copied_count += 1

        if copied_count % 500 == 0:
            print(f"Copied and verified: {copied_count}")

    # Display final split distribution.
    print("\nDataset preparation completed!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total copied images: {copied_count}")

    distribution = (
        df.groupby(["dataset_split", "class"])
        .size()
        .unstack(fill_value=0)
    )

    print("\nOriginal split distribution:")
    print(distribution.to_string())

    return OUTPUT_DIR


# ==================================================
# 5. Script entry point
# ==================================================

if __name__ == "__main__":
    prepare_classification_dataset()
