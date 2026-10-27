"""
Optimized dataset download script for Deepfake Detection pipeline.
Location: D:\\Deepfake Detection\\scripts\\download_genimage_hf.py
"""
import sys
from pathlib import Path

# Project root ko Python path mein inject karein
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import sys
import os
from pathlib import Path

# --- BULLETPROOF PATH INJECTION ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(str(PROJECT_ROOT))

from src.config import DIRS
from datasets import load_dataset


def main() -> None:
    print("Downloading lightweight benchmark subset from Hugging Face...")
    
    # Loading benchmark subset for training
    ds = load_dataset("TheKernel01/Tiny-GenImage", split="train[:15000]")
    
    target_root = DIRS["data_raw_image"] / "GenImage"
    print(f"Mapping images into target directory: {target_root}")
    
    counter = {"real": 0, "fake": 0}

    for idx, item in enumerate(ds):
        img = item["image"]
        label_val = item["label"]  # 0 = real, 1 = fake
        gen_name = item.get("generator", "Stable_Diffusion_v1_4")

        if label_val == 0:
            class_folder = "real"
            gen_folder = "real_photo"
            counter["real"] += 1
        else:
            class_folder = "fake"
            gen_folder = str(gen_name).replace(" ", "_").replace(".", "_")
            counter["fake"] += 1

        out_dir = target_root / class_folder / gen_folder
        out_dir.mkdir(parents=True, exist_ok=True)

        file_path = out_dir / f"img_{idx:05d}.jpg"
        if not file_path.exists():
            img.convert("RGB").save(file_path, quality=95)

    print("\n[SUCCESS] Dataset download and mapping complete!")
    print(f"  Total Real Images Saved : {counter['real']}")
    print(f"  Total Fake Images Saved : {counter['fake']}")
    print(f"  Dataset Root Location   : {target_root}")


if __name__ == "__main__":
    main()