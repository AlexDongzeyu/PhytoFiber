from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd


def extract_fiber_rgb_from_image(
    image_path: Path,
    lower_hsv: tuple[int, int, int] = (35, 30, 20),
    upper_hsv: tuple[int, int, int] = (170, 255, 255),
) -> dict:
    """
    Isolates likely fiber pixels via HSV mask and returns average BGR/RGB channels.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))

    # Remove noise for a more stable mean color estimate.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    selected_pixels = image[mask > 0]
    if selected_pixels.size == 0:
        raise ValueError(
            f"No pixels passed mask in {image_path.name}. Tune HSV thresholds."
        )

    b_mean, g_mean, r_mean = selected_pixels.mean(axis=0)
    return {
        "image_name": image_path.name,
        "pixel_count": int(selected_pixels.shape[0]),
        "R": float(r_mean),
        "G": float(g_mean),
        "B": float(b_mean),
    }


def parse_image_metadata(image_name: str) -> dict:
    """
    Expected naming convention: S01_t12.jpg where:
    - S01 is sample_id
    - t12 is time in hours
    """
    stem = Path(image_name).stem
    match = re.match(r"^([A-Za-z0-9]+)_t(\d+)$", stem)
    if not match:
        return {"sample_id": None, "time_h": None}
    return {"sample_id": match.group(1), "time_h": int(match.group(2))}


def batch_extract_folder(
    image_dir: Path,
    file_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    lower_hsv: tuple[int, int, int] = (35, 30, 20),
    upper_hsv: tuple[int, int, int] = (170, 255, 255),
) -> pd.DataFrame:
    rows = []
    images = sorted(
        [p for p in image_dir.glob("*") if p.suffix.lower() in file_extensions]
    )
    if not images:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    for img_path in images:
        row = extract_fiber_rgb_from_image(
            img_path,
            lower_hsv=lower_hsv,
            upper_hsv=upper_hsv,
        )
        row.update(parse_image_metadata(row["image_name"]))
        rows.append(row)
    return pd.DataFrame(rows)

