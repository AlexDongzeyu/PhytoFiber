from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from PIL.Image import Image as PILImage
from pillow_heif import register_heif_opener
from sklearn.cluster import KMeans

register_heif_opener()


SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".heic", ".heif")


def load_image_rgb(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is not None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with Image.open(image_path) as pil_image:
        converted: PILImage = pil_image.convert("RGB")
        return np.asarray(converted)


def parse_image_metadata(image_name: str) -> dict:
    stem = Path(image_name).stem
    match = re.match(r"^([A-Za-z0-9]+)_t(\d+)$", stem)
    if not match:
        return {"sample_id": None, "time_h": None}
    return {"sample_id": match.group(1), "time_h": int(match.group(2))}


def compute_luminance(rgb_pixels: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb_pixels[:, 0] + 0.7152 * rgb_pixels[:, 1] + 0.0722 * rgb_pixels[:, 2]


def segment_fiber_pixels(
    image_rgb: np.ndarray,
    n_clusters: int = 2,
    random_state: int = 42,
    sample_size: int = 30000,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    flat_pixels = image_rgb.reshape(-1, 3).astype(np.float32)
    if flat_pixels.size == 0:
        raise ValueError("Image contains no pixels.")

    if len(flat_pixels) > sample_size:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(len(flat_pixels), size=sample_size, replace=False)
        fit_pixels = flat_pixels[sample_idx]
    else:
        fit_pixels = flat_pixels

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    model.fit(fit_pixels)
    labels = model.predict(flat_pixels)
    cluster_centers = model.cluster_centers_
    cluster_luminance = compute_luminance(cluster_centers)
    fiber_cluster = int(np.argmin(cluster_luminance))
    background_cluster = int(np.argmax(cluster_luminance))

    mask = labels.reshape(image_rgb.shape[:2]) == fiber_cluster
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    selected_pixels = flat_pixels[mask.reshape(-1)]
    if selected_pixels.size == 0:
        raise ValueError("K-means segmentation did not isolate any fiber pixels.")

    diagnostics = {
        "fiber_cluster": fiber_cluster,
        "background_cluster": background_cluster,
        "fiber_brightness": float(cluster_luminance[fiber_cluster]),
        "background_brightness": float(cluster_luminance[background_cluster]),
        "fiber_fraction": float(mask.mean()),
    }
    return selected_pixels, mask, diagnostics


def extract_fiber_rgb_from_image(image_path: Path) -> dict:
    image_rgb = load_image_rgb(image_path)
    selected_pixels, mask, diagnostics = segment_fiber_pixels(image_rgb)

    channel_means = selected_pixels.mean(axis=0)
    channel_medians = np.median(selected_pixels, axis=0)
    channel_std = selected_pixels.std(axis=0)
    metadata = parse_image_metadata(image_path.name)
    return {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "pixel_count": int(selected_pixels.shape[0]),
        "mask_fraction": float(mask.mean()),
        "R": float(channel_means[0]),
        "G": float(channel_means[1]),
        "B": float(channel_means[2]),
        "R_median": float(channel_medians[0]),
        "G_median": float(channel_medians[1]),
        "B_median": float(channel_medians[2]),
        "R_std": float(channel_std[0]),
        "G_std": float(channel_std[1]),
        "B_std": float(channel_std[2]),
        "fiber_brightness": diagnostics["fiber_brightness"],
        "background_brightness": diagnostics["background_brightness"],
        "fiber_fraction": diagnostics["fiber_fraction"],
        **metadata,
    }


def batch_extract_folder(
    image_dir: Path,
    file_extensions: tuple[str, ...] = SUPPORTED_IMAGE_EXTENSIONS,
    recursive: bool = False,
) -> pd.DataFrame:
    rows = []
    iterator = image_dir.rglob("*") if recursive else image_dir.glob("*")
    images = sorted([p for p in iterator if p.is_file() and p.suffix.lower() in file_extensions])
    if not images:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    for img_path in images:
        row = extract_fiber_rgb_from_image(img_path)
        rows.append(row)
    return pd.DataFrame(rows)


def build_image_inventory(
    image_dir: Path,
    file_extensions: tuple[str, ...] = SUPPORTED_IMAGE_EXTENSIONS,
    recursive: bool = True,
) -> pd.DataFrame:
    iterator = image_dir.rglob("*") if recursive else image_dir.glob("*")
    images = sorted([p for p in iterator if p.is_file() and p.suffix.lower() in file_extensions])
    rows = []
    for path in images:
        rows.append(
            {
                "image_name": path.name,
                "relative_path": str(path.relative_to(image_dir)),
                "extension": path.suffix.lower(),
                **parse_image_metadata(path.name),
            }
        )
    return pd.DataFrame(rows)

