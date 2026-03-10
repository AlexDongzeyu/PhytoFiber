from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from phytofiber_analysis.config import COLOR_DATA_FINAL_CSV, CV_EXTRACTED_CSV, IMAGE_INVENTORY_CSV, IMAGES_RAW_DIR, LEGACY_IMAGE_DIR, SPOILAGE_IMAGE_DIR
from phytofiber_analysis.cv_extraction import SUPPORTED_IMAGE_EXTENSIONS, batch_extract_folder, build_image_inventory
from phytofiber_analysis.io_utils import write_csv


def resolve_image_dir() -> tuple[Path, bool]:
    candidates = [
        (SPOILAGE_IMAGE_DIR, False),
        (IMAGES_RAW_DIR, True),
        (LEGACY_IMAGE_DIR, True),
    ]
    for path, recursive in candidates:
        if not path.exists():
            continue
        iterator = path.rglob("*") if recursive else path.glob("*")
        has_supported_images = any(
            candidate.is_file() and candidate.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            for candidate in iterator
        )
        if has_supported_images:
            return path, recursive
    return SPOILAGE_IMAGE_DIR, False


def main() -> None:
    image_dir, recursive = resolve_image_dir()
    df = batch_extract_folder(image_dir=image_dir, recursive=recursive)
    inventory = build_image_inventory(image_dir=image_dir, recursive=recursive)
    matched = df.dropna(subset=["sample_id", "time_h"]).copy()

    write_csv(inventory, IMAGE_INVENTORY_CSV)
    write_csv(df, COLOR_DATA_FINAL_CSV)
    write_csv(matched if not matched.empty else df, CV_EXTRACTED_CSV)
    print(f"Saved image inventory: {IMAGE_INVENTORY_CSV}")
    print(f"Saved color extraction table: {COLOR_DATA_FINAL_CSV}")
    print(f"Saved spoilage-ready CV table: {CV_EXTRACTED_CSV}")


if __name__ == "__main__":
    main()

