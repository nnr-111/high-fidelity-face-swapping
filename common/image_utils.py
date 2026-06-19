from pathlib import Path

import cv2
import numpy as np


def load_bgr(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def save_bgr(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def resize_like(image, reference):
    if image.shape[:2] == reference.shape[:2]:
        return image
    h, w = reference.shape[:2]
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)


def normalize_to_jpg(input_path, output_path):
    image = load_bgr(input_path)
    save_bgr(output_path, image)
    return str(output_path)


def sharpness_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def l1_drift(image_a, image_b):
    if image_a.shape != image_b.shape:
        image_b = resize_like(image_b, image_a)
    diff = np.abs(image_a.astype(np.float32) - image_b.astype(np.float32))
    return float(np.mean(diff) / 255.0)


def image_residual(enhanced, base):
    enhanced = resize_like(enhanced, base)
    return enhanced.astype(np.float32) / 255.0 - base.astype(np.float32) / 255.0
