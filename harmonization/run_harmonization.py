import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.face_utils import build_face_analyzer
from harmonizer import VisualHarmonizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--insightface_root", default="insightface_func/models")
    return parser.parse_args()


def main():
    args = parse_args()
    face_app = build_face_analyzer(args.insightface_root)
    harmonizer = VisualHarmonizer(face_app)
    output, logs = harmonizer.harmonize(args.input, args.target, args.output)
    print(logs)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
