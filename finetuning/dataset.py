import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class FaceSwapPairs(Dataset):
    def __init__(self, pairs_csv, image_size=224):
        self.data = pd.read_csv(pairs_csv)
        self.image_size = image_size

    def __len__(self):
        return len(self.data)

    def _read_image(self, path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        return image * 2.0 - 1.0

    def __getitem__(self, index):
        row = self.data.iloc[index]
        item = {
            "source": self._read_image(row["source_img"]),
            "target": self._read_image(row["target_img"]),
        }
        if "baseline_img" in row:
            item["baseline"] = self._read_image(row["baseline_img"])
        return item
