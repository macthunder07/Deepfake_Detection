"""
Disk Dataset Scanner, Image Integrity Verifier, and Perceptual dHash Calculator.
Scans directory hierarchies of the form:
  <root>/<dataset_name>/<real|fake>/<generator_name>/<image_file>
Location: D:\\Deepfake Detection\\src\\data\\scanner.py
"""

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.config import LABEL_MAP
from src.data.manifest import assign_grouped_splits, compute_file_sha256, validate_manifest

SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """
    Computes a 64-bit perceptual difference hash (dHash) as a 16-char hex string.
    Used to detect near-duplicate images across splits (e.g., re-saved or resized copies).
    """
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bit_string = "".join("1" if b else "0" for b in diff.flatten())
    return f"{int(bit_string, 2):016x}"


def infer_source_id_from_stem(stem: str) -> str:
    """
    Extracts a canonical source_id from a filename stem so related crops/variants
    stay in the same split.
    Example: 'src_0042_sd14_crop1' -> 'src_0042', 'ILSVRC2012_val_00000293_adm' -> 'ILSVRC2012_val_00000293'
    """
    match = re.match(r"^((?:src|ood_src|vid|img)_\d+|ILSVRC\d+_val_\d+|\d{3,6})", stem)
    if match:
        return match.group(1)
    # Fallback: strip trailing variant suffixes like _fake, _real, _crop0
    cleaned = re.sub(r"_(real|fake|sd\d+|glide|midjourney|crop\d+|frame\d+)$", "", stem, flags=re.IGNORECASE)
    return cleaned if cleaned else stem


def scan_image_dataset_directory(
    root_dir: Path,
    task: str = "image",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Walks `root_dir` expecting structure:
      root_dir / <source_dataset> / <real|fake> / <generator> / <file>
    Verifies every image can be opened and decoded, extracts dimensions,
    SHA-256, and perceptual dHash.
    Returns (raw_dataframe, corrupted_or_skipped_files).
    """
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {root_dir}")

    rows: List[Dict[str, object]] = []
    corrupted_files: List[str] = []
    sample_idx = 1

    all_files = sorted([p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS])

    for file_path in all_files:
        rel_parts = file_path.relative_to(root_dir).parts
        # Expected minimum depth: <source_dataset> / <real|fake> / <generator> / <filename>
        if len(rel_parts) < 4:
            continue

        source_dataset = rel_parts[0]
        label_folder = rel_parts[1].lower()
        generator = rel_parts[2]

        if label_folder not in LABEL_MAP:
            continue

        label_num = LABEL_MAP[label_folder]

        # Verify image header and pixel decoding
        try:
            with Image.open(file_path) as img:
                img.load()
                width, height = img.size
                dhash_hex = compute_dhash(img)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            corrupted_files.append(f"{file_path} ({exc})")
            continue

        sha256_hex = compute_file_sha256(file_path)
        source_id = infer_source_id_from_stem(file_path.stem)

        rows.append(
            {
                "sample_id": f"{task}_{sample_idx:06d}",
                "file_path": str(file_path.resolve()),
                "task": task,
                "label": label_num,
                "label_str": label_folder,
                "source_dataset": source_dataset,
                "generator": generator,
                "source_id": source_id,
                "split": "",
                "sha256": sha256_hex,
                "dhash": dhash_hex,
                "width": int(width),
                "height": int(height),
            }
        )
        sample_idx += 1

    df = pd.DataFrame(rows)
    return df, corrupted_files


def check_perceptual_near_duplicates_across_splits(df: pd.DataFrame) -> List[str]:
    """
    Checks if identical perceptual dHashes appear in more than one split
    (e.g. a re-compressed JPEG copy of a training image placed in validation/test).
    """
    if "dhash" not in df.columns or len(df) == 0:
        return []
    dhash_splits = df.groupby("dhash")["split"].nunique()
    leaked_dhashes = dhash_splits[dhash_splits > 1].index.tolist()
    return leaked_dhashes


def build_and_validate_image_manifest(
    root_dir: Path,
    ood_generators: Optional[Iterable[str]] = None,
    val_ratio: float = 0.15,
    test_id_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, object], List[str]]:
    """
    Full pipeline: scan disk -> assign grouped splits -> validate manifest -> check dHash leakage.
    """
    raw_df, corrupted = scan_image_dataset_directory(root_dir, task="image")
    if len(raw_df) == 0:
        raise RuntimeError(
            f"No valid images found under '{root_dir}'. Ensure folders follow: "
            f"<root>/<dataset_name>/<real|fake>/<generator>/<image_file>"
        )

    split_df = assign_grouped_splits(
        raw_df,
        ood_generators=ood_generators,
        val_ratio=val_ratio,
        test_id_ratio=test_id_ratio,
        seed=seed,
    )

    report = validate_manifest(
        split_df,
        ood_generators=ood_generators,
        check_files_exist=True,
        strict_ood_source_isolation=True,
    )

    leaked_dhashes = check_perceptual_near_duplicates_across_splits(split_df)
    report["corrupted_files_skipped"] = len(corrupted)
    report["near_duplicate_dhashes_across_splits"] = len(leaked_dhashes)

    if leaked_dhashes:
        report["warnings"].append(
            f"Found {len(leaked_dhashes)} perceptual dHash collision(s) across splits: {leaked_dhashes[:5]}"
        )

    return split_df, report, corrupted