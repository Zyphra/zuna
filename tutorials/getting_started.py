#!/usr/bin/env python3
"""
Zuna Complete Pipeline Tutorial

This tutorial runs the complete EEG reconstruction pipeline:
1. Preprocessing: .fif → .pt (filtered, epoched, normalized)
2. Model Inference: .pt → .pt (reconstructed by model)
3. Reconstruction: .pt → .fif (denormalized, continuous)

Simply edit the paths below and run:
    python getting_started.py
"""

from pathlib import Path
from zuna.pipeline import run_zuna

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
INPUT_DIR = "/data/datasets/bci/dataset_downloads_cw/pip_test/1_fif_input"
WORKING_DIR = "/data/datasets/bci/dataset_downloads_cw/pip_test/working"
CHECKPOINT = "/data/checkpoints/bci/bci_AY2l_bigrun16e/checkpoints/0000150000"

# Processing options
TARGET_CHANNEL_COUNT = 40  # None for no upsampling, or target channel count (e.g., 40, 64, 128)
                             # New channels added with zeros for model to interpolate
KEEP_INTERMEDIATE_FILES = True  # If False, deletes .pt files after reconstruction
GPU_DEVICE = 5

# Visualization options
PLOT_PT_COMPARISON = False  # Plot .pt file comparisons (preprocessed vs model output)
PLOT_FIF_COMPARISON = True  # Plot .fif file comparisons (preprocessed vs reconstructed)

# =============================================================================
# RUN PIPELINE
# =============================================================================

if __name__ == "__main__":
    run_zuna(
        input_dir=INPUT_DIR,
        working_dir=WORKING_DIR,
        checkpoint_path=CHECKPOINT,
        target_channel_count=TARGET_CHANNEL_COUNT,
        keep_intermediate_files=KEEP_INTERMEDIATE_FILES,
        gpu_device=GPU_DEVICE,
        plot_pt_comparison=PLOT_PT_COMPARISON,
        plot_fif_comparison=PLOT_FIF_COMPARISON,
    )
