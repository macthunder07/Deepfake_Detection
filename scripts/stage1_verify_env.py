"""
Stage 1: Environment, Workspace, GPU, and Baseline Xception Verification.
Location: D:\\Deepfake Detection\\scripts\\stage1_verify_env.py
"""

import sys
import os
import platform
import importlib.metadata
from pathlib import Path


def print_section(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f" {title}")
    print(f"{'=' * 65}")


def check_workspace_and_python() -> Path:
    print_section("1. WORKSPACE & PYTHON INTERPRETER")
    expected_root = Path(r"D:\Deepfake Detection")
    cwd = Path.cwd()

    print(f"OS Platform       : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Current Directory : {cwd}")
    print(f"Expected Root     : {expected_root}")

    if not expected_root.exists():
        print(f"[WARNING] Expected root '{expected_root}' does not exist on disk.")
        target_dir = cwd
    else:
        target_dir = expected_root

    # List top-level contents without modifying anything
    try:
        items = sorted(target_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        print(f"\nExisting items in '{target_dir}' ({len(items)} total):")
        for item in items:
            kind = "[DIR] " if item.is_dir() else "[FILE]"
            print(f"  {kind} {item.name}")
    except Exception as exc:
        print(f"[ERROR] Could not inspect directory contents: {exc}")

    # Verify virtual environment interpreter
    exe_path = Path(sys.executable).resolve()
    print(f"\nPython Version    : {sys.version.split()[0]}")
    print(f"Python Executable : {exe_path}")

    in_venv = (
        ".venv" in [p.lower() for p in exe_path.parts]
        or (sys.prefix != sys.base_prefix)
    )
    if not in_venv:
        print("[ERROR] Python is NOT running inside the project's .venv virtual environment!")
        sys.exit(1)
    else:
        print("[OK] Running inside virtual environment (.venv verified).")

    return target_dir


def check_installed_libraries() -> None:
    print_section("2. REQUIRED LIBRARIES CHECK")
    # Map import name -> package distribution name for version lookup
    required_packages = {
        "torch": "torch",
        "torchvision": "torchvision",
        "timm": "timm",
        "cv2": "opencv-python",
        "sklearn": "scikit-learn",
        "pandas": "pandas",
        "PIL": "Pillow",
        "streamlit": "streamlit",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
    }

    missing = []
    for module_name, dist_name in required_packages.items():
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", None)
            if not version:
                try:
                    version = importlib.metadata.version(dist_name)
                except importlib.metadata.PackageNotFoundError:
                    version = "installed (version metadata unknown)"
            print(f"  [OK] {dist_name:<18} (import: {module_name:<12}) -> v{version}")
        except ImportError as exc:
            print(f"  [MISSING] {dist_name:<18} (import: {module_name:<12}) -> {exc}")
            missing.append(dist_name)

    if missing:
        print(f"\n[ERROR] Missing required packages: {', '.join(missing)}")
        sys.exit(1)


def check_cuda_and_gpu() -> "torch.device":
    import torch

    print_section("3. PYTORCH, CUDA & GPU HARDWARE CHECK")
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"CUDA Compiled Ver : {torch.version.cuda}")
    print(f"cuDNN Version     : {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available    : {cuda_available}")

    if not cuda_available:
        print("[ERROR] CUDA is not available. Verify driver and PyTorch CUDA build.")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    print(f"GPU Device Count  : {device_count}")

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    total_vram_gb = props.total_memory / (1024 ** 3)
    bf16_supported = torch.cuda.is_bf16_supported()

    print(f"GPU Name          : {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"Total VRAM        : {total_vram_gb:.2f} GB ({props.total_memory / (1024 ** 2):.0f} MB)")
    print(f"BF16 Supported    : {bf16_supported}")

    return device


def check_xception_baseline(device: "torch.device") -> None:
    import torch
    import timm
    from timm.data import resolve_data_config

    print_section("4. TIMM PRETRAINED XCEPTION & MIXED PRECISION TEST")
    print(f"timm Version      : {timm.__version__}")

    # List available xception variants in this timm version
    xception_variants = timm.list_models("*xception*", pretrained=True)
    print(f"Available Pretrained Xception Variants: {xception_variants}")

    # 'legacy_xception' and 'xception' refer to the classic 299x299 Chollet Xception architecture
    model_name = "xception" if "xception" in xception_variants else "legacy_xception"
    print(f"Selected Baseline : '{model_name}' (num_classes=2 -> [0: Real/Authentic, 1: Fake/Manipulated])")

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # Load pretrained weights and replace classification head with 2 classes
        model = timm.create_model(model_name, pretrained=True, num_classes=2)
        model.eval()
        model.to(device)

        data_cfg = resolve_data_config({}, model=model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Model Loaded      : Success")
        print(f"Resolved Data Cfg : {data_cfg}")
        print(f"Total Parameters  : {total_params:,} (Trainable: {trainable_params:,})")

        # Dummy batch of 4 images at 299x299
        dummy_input = torch.randn(4, 3, 299, 299, device=device)

        # 1. FP32 Forward Pass
        with torch.no_grad():
            logits_fp32 = model(dummy_input)
        if not torch.isfinite(logits_fp32).all():
            raise RuntimeError("Non-finite values (NaN/Inf) detected in FP32 logits!")

        # 2. AMP (FP16) Forward Pass
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits_fp16 = model(dummy_input)
        if not torch.isfinite(logits_fp16).all():
            raise RuntimeError("Non-finite values (NaN/Inf) detected in AMP FP16 logits!")

        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        print(f"Dummy Input Shape : {tuple(dummy_input.shape)}")
        print(f"FP32 Output Shape : {tuple(logits_fp32.shape)} | dtype: {logits_fp32.dtype} | Finite: True")
        print(f"FP16 Output Shape : {tuple(logits_fp16.shape)} | dtype: {logits_fp16.dtype} | Finite: True")
        print(f"Peak VRAM Used    : {peak_vram_mb:.2f} MB")
        print("\n[SUCCESS] Stage 1 diagnostics completed with zero errors.")

    except Exception as exc:
        print(f"\n[ERROR] Xception verification failed: {exc}")
        sys.exit(1)


def main() -> None:
    check_workspace_and_python()
    check_installed_libraries()
    device = check_cuda_and_gpu()
    check_xception_baseline(device)


if __name__ == "__main__":
    main()