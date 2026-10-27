"""
PyTorch Dataset, Forensic-Safe Transforms, and Class-Weight Computation.
Location: D:\\Deepfake Detection\\src\\data\\dataset.py
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import ModelConfig


def get_forensic_transforms(
    split: str,
    model_cfg: ModelConfig = ModelConfig(),
) -> transforms.Compose:
    """
    Builds preprocessing transforms for 299x299 Xception without erasing forensic traces.
    - Avoids heavy blur, aggressive color jitter, or synthetic noise injection.
    - Uses Bicubic interpolation to match timm's legacy_xception configuration.
    """
    _, height, width = model_cfg.input_size
    interp = transforms.InterpolationMode.BICUBIC

    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((height, width), interpolation=interp, antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=model_cfg.mean, std=model_cfg.std),
            ]
        )
    else:
        # Deterministic evaluation preprocessing for val, test_id, test_ood
        return transforms.Compose(
            [
                transforms.Resize((height, width), interpolation=interp, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=model_cfg.mean, std=model_cfg.std),
            ]
        )


class DeepfakeImageDataset(Dataset):
    """
    Manifest-backed PyTorch Dataset for AI-Generated Image Detection
    and extracted Video Face-Crop Classification.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        split: str,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()
        self.split = split
        self.df = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No samples found in manifest for split='{split}'.")

        self.transform = transform or get_forensic_transforms(split=split)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.df.iloc[index]
        img_path = Path(str(row["file_path"]))

        if not img_path.is_file():
            raise FileNotFoundError(f"Missing image file during DataLoader fetch: {img_path}")

        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")

        tensor_img = self.transform(rgb_img)
        label = int(row["label"])

        return {
            "image": tensor_img,
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": str(row["sample_id"]),
            "generator": str(row["generator"]),
            "source_dataset": str(row["source_dataset"]),
            "source_id": str(row["source_id"]),
            "file_path": str(img_path),
        }


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    Computes balanced inverse-frequency class weights [w_real, w_fake]
    from the training split to handle class imbalance cleanly.
    """
    counts = train_df["label"].value_counts().to_dict()
    n_real = counts.get(0, 0)
    n_fake = counts.get(1, 0)
    if n_real == 0 or n_fake == 0:
        raise ValueError(f"Training split is missing a class! Counts: real(0)={n_real}, fake(1)={n_fake}")

    total = n_real + n_fake
    w_real = total / (2.0 * n_real)
    w_fake = total / (2.0 * n_fake)
    return torch.tensor([w_real, w_fake], dtype=torch.float32)


def create_image_dataloaders(
    manifest_csv: Path,
    batch_size: int = ModelConfig.default_batch_size,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[Dict[str, DataLoader], torch.Tensor, pd.DataFrame]:
    """
    Loads a validated manifest CSV and returns a dictionary of DataLoaders
    for all available splits ('train', 'val', 'test_id', 'test_ood'),
    plus the training class-weight tensor and full DataFrame.
    """
    if not manifest_csv.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    df = pd.read_csv(manifest_csv)
    loaders: Dict[str, DataLoader] = {}

    for split in ("train", "val", "test_id", "test_ood"):
        sub_df = df[df["split"] == split]
        if len(sub_df) == 0:
            continue

        dataset = DeepfakeImageDataset(df, split=split)
        is_train = split == "train"
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=False,
        )

    train_df = df[df["split"] == "train"]
    class_weights = compute_class_weights(train_df)

    return loaders, class_weights, df