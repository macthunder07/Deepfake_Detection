"""
Streamlit Web Application for Generalizable AI-Generated Image
and Deepfake Video Detection.
Location: D:\\Deepfake Detection\\app.py
"""

import time
import json
from pathlib import Path
import sys
import os

# --- BULLETPROOF PATH RESOLUTION ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(str(PROJECT_ROOT))

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import streamlit as st
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms

# Direct imports from src modules
from src.config import DIRS, ModelConfig, INV_LABEL_MAP
from src.video.face_extractor import extract_faces_from_video
from src.video.aggregator import aggregate_video_scores


# ---------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Deepfake & AI Image Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
.sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
.metric-card { background-color: #F3F4F6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #3B82F6; }
.warning-box { background-color: #FEF3C7; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #F59E0B; color: #92400E; }
.success-box { background-color: #D1FAE5; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #10B981; color: #065F46; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MODEL & CHECKPOINT LOADER (CACHED)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_detection_system():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig()

    model = timm.create_model(cfg.architecture, pretrained=True, num_classes=cfg.num_classes)
    
    best_ckpt_path = DIRS["checkpoints_image"] / "best_model.pth"
    calib_path = DIRS["checkpoints_image"] / "calibration_meta.json"

    calibrated_threshold = 0.35  # Strict AI-detection threshold
    checkpoint_status = "Base Pretrained Xception (No custom training checkpoint found yet)"

    if best_ckpt_path.is_file():
        try:
            ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            checkpoint_status = f"Trained Checkpoint Loaded (Epoch {ckpt.get('epoch', '?')}, Val AUC: {ckpt.get('val_auc', 0.0):.4f})"
        except Exception as exc:
            checkpoint_status = f"Error loading checkpoint: {exc}"

    if calib_path.is_file():
        try:
            with open(calib_path, "r", encoding="utf-8") as f:
                calib_data = json.load(f)
                calibrated_threshold = calib_data.get("calibrated_threshold", 0.5)
        except Exception:
            pass

    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )

    return model, device, transform, calibrated_threshold, checkpoint_status


with st.spinner("Initializing Deepfake Detection System and loading Xception on GPU..."):
    model, device, transform, calibrated_threshold, checkpoint_status = load_detection_system()


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.markdown("### System & Hardware")
    st.write(f"**Device:** `{device}`")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        st.write(f"**GPU:** {gpu_name}")
        st.write(f"**VRAM:** {vram_gb:.2f} GB")
    
    st.markdown("---")
    st.markdown("### Model Configuration")
    st.write(f"**Architecture:** `legacy_xception` (299×299)")
    st.write(f"**Threshold (Youden's J):** `{calibrated_threshold:.4f}`")
    st.markdown(f"<small>{checkpoint_status}</small>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Generalization Notice")
    st.warning(
        "Forensic detectors have inherent uncertainty when evaluating unseen generators "
        "or heavily compressed media. Review confidence scores and uncertainty warnings."
    )


# ---------------------------------------------------------------------------
# MAIN INTERFACE
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🛡️ Generalizable AI-Generated Image & Deepfake Video Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Research-grade forensic detection powered by PyTorch, Xception, and GPU acceleration.</div>', unsafe_allow_html=True)

tab_image, tab_video = st.tabs(["🖼️ AI-Generated Image Detection", "🎥 Deepfake Video Detection"])


# ===========================================================================
# TAB 1: IMAGE DETECTION
# ===========================================================================
with tab_image:
    st.subheader("Task A: AI-Generated Image Classification")
    st.write("Upload a photograph or an AI-generated image (JPG, PNG, WEBP) to classify its origin.")

    uploaded_image = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_image is not None:
        col_img, col_res = st.columns([1, 1], gap="large")

        with col_img:
            pil_img = Image.open(uploaded_image).convert("RGB")
            st.image(pil_img, caption=f"Uploaded Image ({pil_img.size[0]}×{pil_img.size[1]} px)", use_container_width=True)

        with col_res:
            if st.button("Run Image Forensic Analysis", type="primary", use_container_width=True):
                t_start = time.time()
                
                tensor_img = transform(pil_img).unsqueeze(0).to(device)
                amp_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

                with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                    logits = model(tensor_img)
                    probs = torch.softmax(logits, dim=1)[0].float().cpu().numpy()

                elapsed_ms = (time.time() - t_start) * 1000.0

                prob_real = float(probs[0])
                prob_fake = float(probs[1])
                pred_label = 1 if prob_fake >= calibrated_threshold else 0
                label_str = INV_LABEL_MAP[pred_label]

                st.markdown("### Analysis Results")
                
                if label_str == "fake":
                    st.markdown(
                        f'<div class="warning-box"><b>Classification: AI-GENERATED / MANIPULATED</b><br>'
                        f'Fake Probability: <b>{prob_fake * 100:.2f}%</b> (Threshold: {calibrated_threshold:.2f})</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="success-box"><b>Classification: AUTHENTIC PHOTOGRAPH</b><br>'
                        f'Real Probability: <b>{prob_real * 100:.2f}%</b> (Threshold: {calibrated_threshold:.2f})</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(f"**Inference Latency:** `{elapsed_ms:.1f} ms` on `{device}`")
                
                st.write("Score Distribution:")
                st.progress(prob_fake, text=f"AI-Generated Score: {prob_fake:.4f}")
                st.progress(prob_real, text=f"Authentic Score: {prob_real:.4f}")

                margin = abs(prob_fake - calibrated_threshold)
                if margin < 0.15:
                    st.markdown(
                        f"<small>⚠️ **Uncertainty Warning:** Score is close to the decision boundary "
                        f"(margin: {margin:.3f}). Consider inspecting artifacts under higher resolution.</small>",
                        unsafe_allow_html=True,
                    )


# ===========================================================================
# TAB 2: VIDEO DETECTION
# ===========================================================================
with tab_video:
    st.subheader("Task B: Deepfake Video Detection")
    st.write("Upload a video container (MP4, AVI, MOV), sample frames, extract face crops, and compute video-level scores.")

    uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])
    
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        sample_rate = st.slider("Frame Sample Rate (extract every N-th frame)", min_value=5, max_value=60, value=15, step=5)
    with col_cfg2:
        agg_strategy = st.selectbox("Video Score Aggregation Strategy", options=["mean", "max", "top_k"], index=0, help="max = worst-frame detection; mean = average frame score; top_k = average of top 20% frames.")

    if uploaded_video is not None:
        temp_video_path = DIRS["data_raw_video"] / uploaded_video.name
        temp_video_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        st.video(str(temp_video_path))

        if st.button("Run Video Deepfake Forensic Pipeline", type="primary", use_container_width=True):
            with st.spinner("Extracting frames and detecting faces..."):
                t_start = time.time()
                try:
                    extracted_records = extract_faces_from_video(
                        video_path=temp_video_path,
                        output_face_dir=DIRS["data_processed_faces"],
                        sample_rate=sample_rate,
                        target_size=(299, 299),
                    )
                except Exception as exc:
                    st.error(f"Error during face extraction: {exc}")
                    extracted_records = []

            if len(extracted_records) == 0:
                st.error("No valid frames or faces could be extracted from this video file.")
            else:
                with st.spinner(f"Running Xception GPU inference on {len(extracted_records)} sampled frames..."):
                    frame_probs = []
                    amp_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

                    with torch.no_grad():
                        for rec in extracted_records:
                            pil_face = Image.open(rec["file_path"]).convert("RGB")
                            tensor_img = transform(pil_face).unsqueeze(0).to(device)

                            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                                logits = model(tensor_img)
                                prob_fake = float(torch.softmax(logits, dim=1)[0, 1].float().item())

                            frame_probs.append(prob_fake)

                    agg_result = aggregate_video_scores(np.array(frame_probs), strategy=agg_strategy)
                    video_score = agg_result["video_score"]
                    elapsed_total = time.time() - t_start

                    is_manipulated = video_score >= calibrated_threshold

                    st.markdown("### Video Forensic Summary")
                    if is_manipulated:
                        st.markdown(
                            f'<div class="warning-box"><b>Video Classification: LIKELY MANIPULATED (DEEPFAKE)</b><br>'
                            f'Aggregated Video Score ({agg_strategy}): <b>{video_score * 100:.2f}%</b> (Threshold: {calibrated_threshold:.2f})</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="success-box"><b>Video Classification: LIKELY AUTHENTIC</b><br>'
                            f'Aggregated Video Score ({agg_strategy}): <b>{video_score * 100:.2f}%</b> (Threshold: {calibrated_threshold:.2f})</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown(f"**Total Processing Time:** `{elapsed_total:.2f}s` | **Frames Evaluated:** `{len(extracted_records)}`")

                    st.markdown("### Sampled Face Crops & Frame Scores")
                    cols = st.columns(min(len(extracted_records), 4))
                    for idx, rec in enumerate(extracted_records):
                        col_idx = idx % 4
                        with cols[col_idx]:
                            score = frame_probs[idx]
                            badge = "🔴 Fake" if score >= calibrated_threshold else "🟢 Real"
                            st.image(rec["file_path"], caption=f"Frame {rec['frame_index']} | {badge}\n({score:.2f})", use_container_width=True)