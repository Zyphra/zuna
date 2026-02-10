#!/usr/bin/env python3
"""
Zuna Complete Pipeline Tutorial

This tutorial shows two ways to run the pipeline:
1. Complete pipeline (all steps in one call)
2. Individual steps (for more control over intermediate files)

Simply edit the paths below and run:
    python 4_tutorial_complete_pipeline.py
"""

from pathlib import Path
from zuna.pipeline import run_zuna, zuna_preprocessing, zuna_inference, zuna_pt_to_fif

# =============================================================================
# CONFIGURE YOUR PATHS HERE
# =============================================================================

# INPUT_DIR = "data/1_fif_input"                # Input .fif files
# PT_INPUT_DIR = 'data/2_pt_input'     # Where to save preprocessed .pt files (None = OUTPUT_DIR/tmp/pt_input)
# PT_OUTPUT_DIR = 'data/3_pt_output'    # Where to save model output .pt files (None = OUTPUT_DIR/tmp/pt_output)
# OUTPUT_DIR = "data/4_fif_output"              # Output reconstructed .fif files
# CHECKPOINT = "/data/checkpoints/bci/bci_AY2l_bigrun16e/checkpoints/0000150000"

INPUT_DIR = "/data/datasets/bci/dataset_downloads_cw/pip_test/1_fif"                # Input .fif files
PT_INPUT_DIR = '/data/datasets/bci/dataset_downloads_cw/pip_test/2_pt_input'     # Where to save preprocessed .pt files (None = OUTPUT_DIR/tmp/pt_input)
PT_OUTPUT_DIR = '/data/datasets/bci/dataset_downloads_cw/pip_test/3_pt_output'    # Where to save model output .pt files (None = OUTPUT_DIR/tmp/pt_output)
OUTPUT_DIR = "/data/datasets/bci/dataset_downloads_cw/pip_test/4_fif_output"              # Output reconstructed .fif files
CHECKPOINT = "/data/checkpoints/bci/bci_AY2l_bigrun16e/checkpoints/0000150000"

UPSAMPLE_FACTOR = None  # None for no upsampling, or integer (e.g., 2, 4)
GPU_DEVICE = 1          # GPU device ID

# =============================================================================
# OPTION 1: Complete Pipeline (Recommended)
# =============================================================================
# Runs all 3 steps automatically and cleans up tmp files

if __name__ == "__main__":
    print("Running complete pipeline...")
    run_zuna(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        checkpoint_path=CHECKPOINT,
        upsample_factor=UPSAMPLE_FACTOR,
        pt_input_dir=PT_INPUT_DIR,
        pt_output_dir=PT_OUTPUT_DIR,
        gpu_device=GPU_DEVICE,
    )

# =============================================================================
# OPTION 2: Individual Steps (For Advanced Users)
# =============================================================================
# Run each step separately for more control

if __name__ == "__main__" and False:  # Change False to True to use this option
    print("Running individual steps...")

    # Setup paths for intermediate files
    output_path = Path(OUTPUT_DIR)
    pt_input = PT_INPUT_DIR if PT_INPUT_DIR else str(output_path / "tmp" / "pt_input")
    pt_output = PT_OUTPUT_DIR if PT_OUTPUT_DIR else str(output_path / "tmp" / "pt_output")

    # Create directories
    Path(pt_input).mkdir(parents=True, exist_ok=True)
    Path(pt_output).mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Preprocessing (.fif → .pt)
    print("\n" + "="*80)
    print("STEP 1: Preprocessing")
    print("="*80)
    zuna_preprocessing(
        input_dir=INPUT_DIR,
        output_dir=pt_input,
        target_sfreq=256.0,
        epoch_duration=5.0,
        apply_notch_filter=False
    )

    # Step 2: Model Inference (.pt → .pt)
    print("\n" + "="*80)
    print("STEP 2: Model Inference")
    print("="*80)
    zuna_inference(
        input_dir=pt_input,
        output_dir=pt_output,
        checkpoint_path=CHECKPOINT,
        gpu_device=GPU_DEVICE
    )

    # Step 3: Reverse (.pt → .fif)
    print("\n" + "="*80)
    print("STEP 3: Reverse (.pt → .fif)")
    print("="*80)
    zuna_pt_to_fif(
        input_dir=pt_output,
        output_dir=OUTPUT_DIR,
        upsample_factor=UPSAMPLE_FACTOR
    )

    # Optional: Clean up tmp directories if they were auto-created
    if PT_INPUT_DIR is None and PT_OUTPUT_DIR is None:
        import shutil
        tmp_path = output_path / "tmp"
        if tmp_path.exists():
            print("\nCleaning up temporary files...")
            shutil.rmtree(tmp_path)
            print(f"✓ Removed: {tmp_path}")

    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE!")
    print("="*80)
    print(f"Final output: {OUTPUT_DIR}")
    print("="*80)
