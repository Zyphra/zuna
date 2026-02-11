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
# TARGET_CHANNEL_COUNT = None   # no upsampling
# TARGET_CHANNEL_COUNT = 40     # upsample to N channels (greedy selection)
TARGET_CHANNEL_COUNT = ['AF3', 'AF4', 'F1', 'F2', 'FC1', 'FC2', 'CP1', 'CP2', 'PO3', 'PO4'] # add specific channels from 10-05 montage

KEEP_INTERMEDIATE_FILES = True  # If False, deletes .pt files after reconstruction
GPU_DEVICE = 5

# Visualization options
PLOT_PT_COMPARISON = False  # Plot .pt file comparisons (preprocessed vs model output)
PLOT_FIF_COMPARISON = True  # Plot .fif file comparisons (preprocessed vs reconstructed)

# =============================================================================
# RUN PIPELINE
# =============================================================================

if __name__ == "__main__":
    if False: 
        # Option 1: Run complete pipeline (recommended)
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


    # =============================================================================
    # Option 2: Run steps individually (uncomment to use)
    # =============================================================================
    if True: 

        from zuna.pipeline import zuna_step1_preprocess, zuna_step2_inference, zuna_step3_reconstruct

        # Step 1: Preprocessing (.fif → .pt)
        zuna_step1_preprocess(
            input_dir=INPUT_DIR,
            working_dir=WORKING_DIR,
            target_channel_count=TARGET_CHANNEL_COUNT,
        )

        # Step 2: Model Inference (.pt → .pt)
        zuna_step2_inference(
            working_dir=WORKING_DIR,
            checkpoint_path=CHECKPOINT,
            gpu_device=GPU_DEVICE,
        )

        # Step 3: Reconstruction (.pt → .fif)
        zuna_step3_reconstruct(
            working_dir=WORKING_DIR,
        )
