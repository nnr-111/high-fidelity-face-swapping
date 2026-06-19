## Structure

```text
high-fidelity-face-swapping/
├── common/
│   ├── face_utils.py
│   └── image_utils.py
├── finetuning/
│   ├── dataset.py
│   ├── losses.py
│   └── train.py
├── enhancement/
│   ├── hf_enhancer.py
│   └── run_enhancement.py
├── harmonization/
│   ├── harmonizer.py
│   └── run_harmonization.py
├── demo/
│   └── gradio_demo.py
├── configs/
│   ├── finetune.yaml
│   └── paths.yaml
├── requirements.txt
└── README.md
```

## Methods

### Partial Fine-Tuning

Fine-tunes selected identity-related SimSwap parameters while keeping most of the generator stable. The objective is to improve source identity preservation without strongly changing pose, expression, and background.

Run:

```bash
python finetuning/train.py   --config configs/finetune.yaml   --pairs_csv data/pairs.csv   --output_dir outputs/finetune
```

### High-Frequency Enhancement

Uses Real-ESRGAN to obtain detail residuals, then injects only controlled high-frequency information. Candidates are filtered using identity similarity, visual drift, and sharpness gain.

Run:

```bash
python enhancement/run_enhancement.py   --source data/source.jpg   --input outputs/simswap/result.jpg   --output outputs/enhancement/enhanced.jpg   --realesrgan_root /path/to/Real-ESRGAN
```

### Visual Harmonization

Matches skin tone and lighting between the enhanced result and target image, then blends the face region with a soft mask.

Run:

```bash
python harmonization/run_harmonization.py   --input outputs/enhancement/enhanced.jpg   --target data/target.jpg   --output outputs/harmonization/final.jpg
```

### Demo

Runs the full inference demo using selected image pairs.

```bash
python demo/gradio_demo.py   --pairs_csv data/pairs.csv   --simswap_root /path/to/SimSwap   --realesrgan_root /path/to/Real-ESRGAN
```

## Notes

Model checkpoints are not included. Put the required SimSwap, ArcFace, InsightFace, and Real-ESRGAN weights in their original folders and update the paths.
