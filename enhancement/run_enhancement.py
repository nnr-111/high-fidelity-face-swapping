import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.face_utils import build_face_analyzer
from hf_enhancer import HighFrequencyEnhancer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--realesrgan_root", required=True)
    parser.add_argument("--insightface_root", default="insightface_func/models")
    parser.add_argument("--work_dir", default="outputs/enhancement/work")
    return parser.parse_args()


def main():
    args = parse_args()
    face_app = build_face_analyzer(args.insightface_root)
    enhancer = HighFrequencyEnhancer(face_app, args.realesrgan_root)
    output, logs = enhancer.enhance(args.source, args.input, args.output, args.work_dir)
    print(logs)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
