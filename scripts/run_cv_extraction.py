from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from phytofiber_analysis.config import DATA_RAW_DIR, CV_EXTRACTED_CSV
from phytofiber_analysis.cv_extraction import batch_extract_folder
from phytofiber_analysis.io_utils import write_csv


def main() -> None:
    image_dir = DATA_RAW_DIR / "spoilage_images"
    df = batch_extract_folder(image_dir=image_dir)
    write_csv(df, CV_EXTRACTED_CSV)
    print(f"Saved CV extraction output: {CV_EXTRACTED_CSV}")


if __name__ == "__main__":
    main()

