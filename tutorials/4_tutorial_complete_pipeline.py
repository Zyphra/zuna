#!/usr/bin/env python3
"""
Minimal Zuna Complete Pipeline

One-line execution: .fif → model inference → reconstructed .fif

Simply edit the paths below and run:
    python 4_tutorial_complete_pipeline.py
"""

from zuna.pipeline import run_zuna

# =============================================================================
# CONFIGURE YOUR PATHS HERE
# =============================================================================

INPUT_DIR = "data/1_fif_input"                # Input .fif files
OUTPUT_DIR = "data/5_fif_output"              # Output reconstructed .fif files
CHECKPOINT = "/data/checkpoints/bci/bci_AY2l_bigrun16e/checkpoints/0000150000"

UPSAMPLE_FACTOR = None  # None for no upsampling, or integer (e.g., 2, 4)
GPU_DEVICE = 1          # GPU device ID

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    run_zuna(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        checkpoint_path=CHECKPOINT,
        upsample_factor=UPSAMPLE_FACTOR,
        gpu_device=GPU_DEVICE,
    )
