import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import gradio as gr
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.face_utils import build_face_analyzer
from common.image_utils import load_bgr, normalize_to_jpg, save_bgr
from enhancement.hf_enhancer import HighFrequencyEnhancer
from harmonization.harmonizer import VisualHarmonizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_csv", required=True)
    parser.add_argument("--simswap_root", required=True)
    parser.add_argument("--realesrgan_root", required=True)
    parser.add_argument("--simswap_model_name", default="people_ft3")
    parser.add_argument("--arcface_checkpoint", default="arcface_model/arcface_checkpoint.tar")
    parser.add_argument("--insightface_root", default="insightface_func/models")
    parser.add_argument("--run_root", default="outputs/demo_runs")
    return parser.parse_args()


args = parse_args()
simswap_root = Path(args.simswap_root)
run_root = Path(args.run_root)
run_root.mkdir(parents=True, exist_ok=True)
pairs = pd.read_csv(args.pairs_csv, dtype={"pair_id": str})
pairs["pair_id"] = pairs["pair_id"].astype(str).str.zfill(6)
choices = pairs["pair_id"].tolist()
face_app = build_face_analyzer(simswap_root / args.insightface_root)


def resolve_path(value):
    path = Path(str(value))
    if path.exists():
        return path
    return simswap_root / path


def get_pair(pair_id):
    pair_id = str(pair_id).zfill(6)
    row = pairs[pairs["pair_id"] == pair_id]
    if len(row) == 0:
        raise gr.Error(f"pair_id not found: {pair_id}")
    row = row.iloc[0]
    return resolve_path(row["source_img"]), resolve_path(row["target_img"])


def preview(pair_id):
    source, target = get_pair(pair_id)
    return str(source), str(target)


def find_latest_image(folder, start_time):
    candidates = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        candidates.extend(Path(folder).rglob(ext))
    candidates = [p for p in candidates if p.stat().st_mtime >= start_time - 2]
    if not candidates:
        raise RuntimeError("SimSwap output was not found.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_simswap(source, target, run_dir):
    out_dir = run_dir / "simswap"
    out_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    command = [sys.executable, "test_one_image.py", "--name", args.simswap_model_name, "--Arc_path", str(simswap_root / args.arcface_checkpoint), "--pic_a_path", str(source), "--pic_b_path", str(target), "--output_path", str(out_dir) + "/"]
    result = subprocess.run(command, cwd=str(simswap_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stdout)
    selected = find_latest_image(out_dir, start_time)
    output = run_dir / "simswap.jpg"
    shutil.copy(selected, output)
    return output, result.stdout


def add_watermark(input_path, output_path, text="GROUP 6"):
    image = load_bgr(input_path)
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, min(w, h) / 512.0 * 0.45)
    thickness = max(1, int(round(min(w, h) / 512.0 * 1.5)))
    margin = max(6, int(min(w, h) * 0.025))
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = w - tw - margin
    y = h - margin
    overlay = image.copy()
    cv2.rectangle(overlay, (max(0, x - 8), max(0, y - th - 8)), (min(w - 1, x + tw + 8), min(h - 1, y + baseline + 8)), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.55, image, 0.45, 0)
    cv2.putText(image, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    save_bgr(output_path, image)
    return str(output_path)


def run(pair_id):
    source_original, target_original = get_pair(pair_id)
    run_dir = run_root / f"pair_{str(pair_id).zfill(6)}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    source = run_dir / "source.jpg"
    target = run_dir / "target.jpg"
    normalize_to_jpg(source_original, source)
    normalize_to_jpg(target_original, target)
    logs = []
    simswap, simswap_log = run_simswap(source, target, run_dir)
    logs.append(simswap_log)
    enhancer = HighFrequencyEnhancer(face_app, args.realesrgan_root)
    enhanced, enhancement_log = enhancer.enhance(source, simswap, run_dir / "enhanced.jpg", run_dir / "enhancement_work")
    logs.append(enhancement_log)
    harmonizer = VisualHarmonizer(face_app)
    final, harmonization_log = harmonizer.harmonize(enhanced, target, run_dir / "final.jpg")
    logs.append(harmonization_log)
    watermarked = add_watermark(final, run_dir / "final_watermark.jpg")
    return str(source), str(target), str(simswap), str(enhanced), str(final), str(watermarked), "

".join(logs)


with gr.Blocks(title="High-Fidelity Face Swapping") as demo:
    gr.Markdown("# High-Fidelity Face Swapping")
    pair_id = gr.Dropdown(choices=choices, value=choices[0], label="Image ID")
    with gr.Row():
        source_preview = gr.Image(label="Source", type="filepath")
        target_preview = gr.Image(label="Target", type="filepath")
    with gr.Row():
        preview_button = gr.Button("Preview")
        run_button = gr.Button("Run", variant="primary")
    with gr.Row():
        simswap_output = gr.Image(label="SimSwap", type="filepath")
        enhanced_output = gr.Image(label="Enhanced", type="filepath")
    with gr.Row():
        final_output = gr.Image(label="Harmonized", type="filepath")
        watermarked_output = gr.Image(label="Watermarked", type="filepath")
    log_box = gr.Textbox(label="Log", lines=20)
    preview_button.click(preview, [pair_id], [source_preview, target_preview])
    pair_id.change(preview, [pair_id], [source_preview, target_preview])
    run_button.click(run, [pair_id], [source_preview, target_preview, simswap_output, enhanced_output, final_output, watermarked_output, log_box])


if __name__ == "__main__":
    demo.queue(max_size=2)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
