import cv2
import numpy as np

from common.face_utils import detect_largest_face
from common.image_utils import l1_drift, load_bgr, resize_like, save_bgr


class VisualHarmonizer:
    def __init__(self, face_app, color_strength=0.35, light_strength=0.25, mask_expand=1.25, blur_ksize=41):
        self.face_app = face_app
        self.color_strength = color_strength
        self.light_strength = light_strength
        self.mask_expand = mask_expand
        self.blur_ksize = blur_ksize

    def create_face_mask(self, image_shape, face):
        h, w = image_shape[:2]
        x1, y1, x2, y2 = face.bbox.astype(np.int32)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = (x2 - x1) * self.mask_expand
        bh = (y2 - y1) * self.mask_expand
        nx1 = int(max(0, cx - bw / 2))
        ny1 = int(max(0, cy - bh / 2))
        nx2 = int(min(w - 1, cx + bw / 2))
        ny2 = int(min(h - 1, cy + bh / 2))
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (int(cx), int(cy))
        axes = (max(1, int((nx2 - nx1) / 2)), max(1, int((ny2 - ny1) / 2)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        return mask

    def feather_mask(self, mask):
        ksize = self.blur_ksize + 1 if self.blur_ksize % 2 == 0 else self.blur_ksize
        soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (ksize, ksize), 0)
        return np.clip(soft, 0.0, 1.0)

    def match_lab_color(self, source, reference, mask):
        mask_bool = mask > 0
        if mask_bool.sum() < 50:
            return source.copy()
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
        source_pixels = source_lab[mask_bool]
        reference_pixels = reference_lab[mask_bool]
        source_mean = source_pixels.mean(axis=0)
        source_std = source_pixels.std(axis=0) + 1e-6
        reference_mean = reference_pixels.mean(axis=0)
        reference_std = reference_pixels.std(axis=0) + 1e-6
        matched_pixels = (source_pixels - source_mean) / source_std * reference_std + reference_mean
        matched_pixels = (1.0 - self.color_strength) * source_pixels + self.color_strength * matched_pixels
        matched = source_lab.copy()
        matched[mask_bool] = matched_pixels
        matched = np.clip(matched, 0, 255).astype(np.uint8)
        return cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)

    def match_luminance(self, source, reference, mask):
        mask_bool = mask > 0
        if mask_bool.sum() < 50:
            return source.copy()
        source_ycrcb = cv2.cvtColor(source, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        reference_ycrcb = cv2.cvtColor(reference, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        source_y = source_ycrcb[:, :, 0]
        reference_y = reference_ycrcb[:, :, 0]
        source_mean = source_y[mask_bool].mean()
        source_std = source_y[mask_bool].std() + 1e-6
        reference_mean = reference_y[mask_bool].mean()
        reference_std = reference_y[mask_bool].std() + 1e-6
        matched_y = (source_y - source_mean) / source_std * reference_std + reference_mean
        source_ycrcb[:, :, 0] = np.clip((1.0 - self.light_strength) * source_y + self.light_strength * matched_y, 0, 255)
        return cv2.cvtColor(source_ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    def harmonize(self, input_path, target_path, output_path):
        enhanced = load_bgr(input_path)
        target = resize_like(load_bgr(target_path), enhanced)
        face = detect_largest_face(self.face_app, enhanced)
        if face is None:
            save_bgr(output_path, enhanced)
            return str(output_path), "fallback=no_face_detected"
        mask = self.create_face_mask(enhanced.shape, face)
        color_matched = self.match_lab_color(enhanced, target, mask)
        light_matched = self.match_luminance(color_matched, target, mask)
        soft = self.feather_mask(mask)[:, :, None]
        output = light_matched.astype(np.float32) * soft + target.astype(np.float32) * (1.0 - soft)
        output = np.clip(output, 0, 255).astype(np.uint8)
        save_bgr(output_path, output)
        logs = ["status=applied", f"color_strength={self.color_strength}", f"light_strength={self.light_strength}", f"mask_expand={self.mask_expand}", f"blur_ksize={self.blur_ksize}", f"drift={l1_drift(output, enhanced):.6f}"]
        return str(output_path), "
".join(logs)
