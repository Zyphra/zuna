#!/usr/bin/env python3
"""
Zuna Pipeline

Runs the complete EEG reconstruction pipeline:
1. Preprocessing: .fif → .pt (filtered, epoched, normalized)
2. Inference: .pt → .pt (reconstructed by model)
3. Reconstruction: .pt → .fif (denormalized, continuous)
4. Visualization: comparison plots (optional)

Edit the paths and options below, then run:
    python run_zuna_pipeline.py

For documentation on each function, run:
    help(zuna.preprocessing)
    help(zuna.inference)
    help(zuna.pt_to_fif)
    help(zuna.compare_plot_pipeline)
"""

import shutil
from pathlib import Path
from zuna import preprocessing, inference, pt_to_fif, compare_plot_pipeline

# =============================================================================
# PATHS
# =============================================================================

TUTORIAL_DIR = Path(__file__).parent.resolve()
INPUT_DIR = str(TUTORIAL_DIR / "data" / "1_fif_input")    ### replace with your path
WORKING_DIR = str(TUTORIAL_DIR / "data" / "working")      ### replace with your path

# Derived paths (pipeline directory structure)
WORKING_PATH = Path(WORKING_DIR)
PREPROCESSED_FIF_DIR = str(WORKING_PATH / "1_fif_filter")
PT_INPUT_DIR = str(WORKING_PATH / "2_pt_input")
PT_OUTPUT_DIR = str(WORKING_PATH / "3_pt_output")
FIF_OUTPUT_DIR = str(WORKING_PATH / "4_fif_output")
FIGURES_DIR = str(WORKING_PATH / "FIGURES")

# =============================================================================
# PREPROCESSING OPTIONS
# =============================================================================

APPLY_NOTCH_FILTER = False      # Automatic notch filter for line noise
APPLY_HIGHPASS_FILTER = True    # 0.5 Hz highpass filter
APPLY_AVERAGE_REFERENCE = True  # Average reference

# Channel options
# TARGET_CHANNEL_COUNT = None   # No upsampling (keep original channels)
# TARGET_CHANNEL_COUNT = 40     # Upsample to N channels (greedy selection)
TARGET_CHANNEL_COUNT = ['AF3', 'AF4', 'F1', 'F2', 'FC1', 'FC2', 'CP1', 'CP2', 'PO3', 'PO4']
BAD_CHANNELS = ['Fz', 'Cz']    # Channels to zero out (set to None to disable)

# Artifact removal (disabled by default)
DROP_BAD_CHANNELS = False       # Detect and remove bad channels
DROP_BAD_EPOCHS = False         # Detect and remove bad epochs
ZERO_OUT_ARTIFACTS = False      # Zero out artifact samples

# =============================================================================
# INFERENCE OPTIONS
# =============================================================================

GPU_DEVICE = 0                  # GPU ID (default: 0) or "" for CPU
TOKENS_PER_BATCH = 100000       # Number of tokens per batch - Increase this number for higher GPU utilization.
DATA_NORM = 10.0                # Data normalization factor denominator 
                                # (NOTE: ZUNA was trained on and expects eeg data to have std = 0.1)

DIFFUSION_CFG = 1.0             # Diffusion process in .sample - Default is 1.0 (i.e., no cfg)
DIFFUSION_SAMPLE_STEPS = 50     # Number of steps in the diffusion process

PLOT_EEG_SIGNAL_SAMPLES = True  # Plot raw eeg for data and model reconstruction for single samples inside inference code.
                                # NOTE: Will use GPU very inefficiently if True. Set to False when running at scale

# =============================================================================
# OUTPUT OPTIONS
# =============================================================================

PLOT_PT_COMPARISON = True       # Plot .pt file comparisons
PLOT_FIF_COMPARISON = True      # Plot .fif file comparisons
KEEP_INTERMEDIATE_FILES = True  # If False, deletes .pt files after reconstruction

NUM_SAMPLES = 2

# =============================================================================
# RUN PIPELINE
# =============================================================================

if __name__ == "__main__":

    # Create working directories
    for d in [PREPROCESSED_FIF_DIR, PT_INPUT_DIR, PT_OUTPUT_DIR, FIF_OUTPUT_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Step 1: Preprocessing (.fif → .pt)
    print("[1/4] Preprocessing...", flush=True)
    preprocessing(
        input_dir=INPUT_DIR,
        output_dir=PT_INPUT_DIR,
        apply_notch_filter=APPLY_NOTCH_FILTER,
        apply_highpass_filter=APPLY_HIGHPASS_FILTER,
        apply_average_reference=APPLY_AVERAGE_REFERENCE,
        preprocessed_fif_dir=PREPROCESSED_FIF_DIR,
        drop_bad_channels=DROP_BAD_CHANNELS,
        drop_bad_epochs=DROP_BAD_EPOCHS,
        zero_out_artifacts=ZERO_OUT_ARTIFACTS,
        target_channel_count=TARGET_CHANNEL_COUNT,
        bad_channels=BAD_CHANNELS,
    )

    # Step 2: Model Inference (.pt → .pt)
    print("[2/4] Model inference...", flush=True)
    inference(
        input_dir=PT_INPUT_DIR,
        output_dir=PT_OUTPUT_DIR,
        gpu_device=GPU_DEVICE,
        tokens_per_batch=TOKENS_PER_BATCH,
        data_norm=DATA_NORM,
        diffusion_cfg=DIFFUSION_CFG,
        diffusion_sample_steps=DIFFUSION_SAMPLE_STEPS,
        plot_eeg_signal_samples=PLOT_EEG_SIGNAL_SAMPLES,
        inference_figures_dir=FIGURES_DIR,
    )

    # Step 3: Reconstruction (.pt → .fif)
    print("[3/4] Reconstructing FIF files...", flush=True)
    pt_to_fif(
        input_dir=PT_OUTPUT_DIR,
        output_dir=FIF_OUTPUT_DIR,
    )

    # Step 4: Visualization (optional)
    if PLOT_PT_COMPARISON or PLOT_FIF_COMPARISON:
        print("[4/4] Visualizing pipeline outputs...", flush=True)
        compare_plot_pipeline(
            input_dir=INPUT_DIR,
            fif_input_dir=PREPROCESSED_FIF_DIR,
            fif_output_dir=FIF_OUTPUT_DIR,
            pt_input_dir=PT_INPUT_DIR,
            pt_output_dir=PT_OUTPUT_DIR,
            output_dir=FIGURES_DIR,
            plot_pt=PLOT_PT_COMPARISON,
            plot_fif=PLOT_FIF_COMPARISON,
            num_samples=NUM_SAMPLES,
        )

    # Cleanup intermediate files if requested
    if not KEEP_INTERMEDIATE_FILES:
        shutil.rmtree(PT_INPUT_DIR)
        shutil.rmtree(PT_OUTPUT_DIR)
        print("Removed intermediate PT files.")

    print("Pipeline complete. Output:", FIF_OUTPUT_DIR)
