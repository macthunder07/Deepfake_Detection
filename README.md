# 🛡️ Generalizable AI-Generated Image & Deepfake Video Detection Pipeline

A modular, production-grade computer vision pipeline designed to detect AI-generated images and deepfake videos across diverse generative models (Midjourney, Stable Diffusion, Flux, etc.). Built using PyTorch, `timm` (Xception architecture), and optimized for NVIDIA GPU acceleration with `bfloat16` mixed precision.

---

## 🚀 Key Features

- **Strict Data Governance:** Implements cryptographic SHA-256 perceptual hashing to eliminate duplicates and strictly prevent data leakage across `train`, `val`, `test_id`, and `test_ood` splits.
- **Calibrated Classification:** Automated decision threshold tuning using Youden's J statistic on validation probabilities to maximize accuracy and minimize false positives.
- **Dual-Mode Streamlit Web App (`app.py`):**
  - **Task A (Image Forensics):** Instant inference on single images with probability scores and forensic confidence breakdowns.
  - **Task B (Video Deepfakes):** Automated face extraction via OpenCV, frame sampling, and multi-frame score aggregation (`mean`, `max`, `top_k`).
- **Hardware Optimized:** Native CUDA support leveraging PyTorch Automatic Mixed Precision (`bfloat16`) for high-speed inference.

---

## 🛠️ Tech Stack
- **Core Framework:** PyTorch, Torchvision
- **Model Architecture:** Xception (`timm`)
- **Web Interface:** Streamlit
- **Computer Vision & Utilities:** OpenCV, NumPy, Pillow, Pandas, Datasets

---

## 📂 Project Structure
```text
Deepfake_Detection/
│
├── app.py                     # Streamlit Interactive Web Application
├── src/
│   ├── config.py              # Global paths and hyperparameters
│   ├── models/                # Model definition & checkpoints wrapper
│   └── video/                 # Face extraction & multi-frame score aggregator
├── scripts/
│   ├── download_genimage_hf.py# Dataset ingestion pipeline from Hugging Face
│   ├── build_real_manifest.py # Anti-leakage manifest generator & SHA-256 checker
│   └── stage4_train_image_detector.py # Model training & threshold calibration
└── outputs/
    ├── checkpoints/           # Saved model weights (best_model.pth)
    └── evaluation/            # Evaluation metrics & confusion matrices
