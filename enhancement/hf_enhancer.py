import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from common.face_utils import cosine_similarity, get_face_embedding
from common.image_utils import image_residual, l1_drift, load_bgr, save_bgr, sharpness_laplacian


class HighFrequencyEnhancer:
    def __init__(self, face_app, realesrgan_root, dn_values=(0.5,), alpha_values=(0.25, 0.5, 0.75, 1.0), sigma_values=(0.8, 1.2), id_eps=0.001, drift_threshold=0.008, sharp_gain_threshold=0.5):
        self.face_app = face_app
        self.realesrgan_root = Path(realesrgan_root)
        self.dn_values = dn_values
        self.alpha_values = alpha_values
        self.sigma_values = sigma_values
        self.id_eps = id_eps
        self.drift_threshold = drift_threshold
        self.sharp_gain_threshold = sharp_gain_threshold

    def run_realesrgan(self, input_path, output_dir, dn):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        command = [sys.executable, str(self.realesrgan_root / "inference_realesrgan.py"), "-n", "realesr-general-x4v3", "-i", str(input_path), "-o", str(output_dir), "--suffix", "out", "--ext", "jpg", "-dn", str(dn)]
        result = subprocess.run(command, cwd=str(self.realesrgan_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stdout)
        candidates = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            candidates.extend(output_dir.rglob(ext))
        candidates = [p for p in candidates if p.stat().st_mtime >= start_time - 2]
        if not candidates:
            raise RuntimeError("Real-ESRGAN output was not found.")
        return max(candidates, key=lambda p: p.stat().st_mtime), result.stdout

    def enhance(self, source_path, input_path, output_path, work_dir):
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        source = load_bgr(source_path)
        base = load_bgr(input_path)
        source_embedding = get_face_embedding(self.face_app, source)
        base_embedding = get_face_embedding(self.face_app, base)
        if source_embedding is None:
            raise RuntimeError("Source face embedding was not detected.")
        base_id = -1.0 if base_embedding is None else cosine_similarity(base_embedding, source_embedding)
        base_sharpness = sharpness_laplacian(base)
        best_image = base.copy()
        best_info = {"status": "base", "id_similarity": base_id, "sharpness": base_sharpness, "drift": 0.0, "dn": None, "alpha": None, "sigma": None}
        logs = [f"base_id_similarity={base_id:.6f}", f"base_sharpness={base_sharpness:.6f}"]
        for dn in self.dn_values:
            realesrgan_output, _ = self.run_realesrgan(input_path, work_dir / f"realesrgan_dn_{dn}", dn)
            restored = load_bgr(realesrgan_output)
            if restored.shape != base.shape:
                restored = cv2.resize(restored, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)
            residual = image_residual(restored, base)
            for sigma in self.sigma_values:
                blur = cv2.GaussianBlur(residual, (0, 0), sigmaX=sigma, sigmaY=sigma)
                residual_hf = residual - blur
                for alpha in self.alpha_values:
                    candidate_float = np.clip(base.astype(np.float32) / 255.0 + alpha * residual_hf, 0.0, 1.0)
                    candidate = np.clip(candidate_float * 255.0, 0, 255).astype(np.uint8)
                    candidate_embedding = get_face_embedding(self.face_app, candidate)
                    candidate_id = base_id if candidate_embedding is None else cosine_similarity(candidate_embedding, source_embedding)
                    candidate_sharpness = sharpness_laplacian(candidate)
                    candidate_drift = l1_drift(candidate, base)
                    sharp_gain = candidate_sharpness - base_sharpness
                    accepted = True
                    if base_embedding is not None and candidate_id < base_id - self.id_eps:
                        accepted = False
                    if candidate_drift > self.drift_threshold:
                        accepted = False
                    if sharp_gain < self.sharp_gain_threshold:
                        accepted = False
                    logs.append(f"dn={dn} alpha={alpha} sigma={sigma} id={candidate_id:.6f} sharp_gain={sharp_gain:.3f} drift={candidate_drift:.6f} accepted={accepted}")
                    if not accepted:
                        continue
                    score = (candidate_id - base_id) * 1000.0 + sharp_gain * 0.001 - candidate_drift * 10.0
                    best_score = (best_info["id_similarity"] - base_id) * 1000.0 + (best_info["sharpness"] - base_sharpness) * 0.001 - best_info["drift"] * 10.0
                    if best_info["status"] == "base" or score > best_score:
                        best_image = candidate.copy()
                        best_info = {"status": "enhanced", "id_similarity": candidate_id, "sharpness": candidate_sharpness, "drift": candidate_drift, "dn": dn, "alpha": alpha, "sigma": sigma}
        save_bgr(output_path, best_image)
        logs.append(f"selected={best_info}")
        return str(output_path), "
".join(logs)
