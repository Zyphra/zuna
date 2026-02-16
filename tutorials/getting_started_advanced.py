#!/usr/bin/env python3
"""
Zuna Advanced Tutorial — Step-by-Step Pipeline

This tutorial runs each pipeline step individually, giving you full
control over intermediate results. Use this if you want to:
- Re-run a single step without repeating earlier ones
- Inspect intermediate .pt files between steps
- Customize preprocessing or visualization separately

For the simple one-line version, see getting_started.py.

For documentation on each function, run:
    help(zuna.zuna_preprocessing)
    help(zuna.zuna_inference)
    help(zuna.zuna_pt_to_fif)
    help(zuna.zuna_plot)
"""

from pathlib import Path
from zuna.pipeline import zuna_preprocessing, zuna_inference, zuna_pt_to_fif, zuna_plot

# =============================================================================
# PATHS
# =============================================================================

TUTORIAL_DIR = Path(__file__).parent.resolve()
INPUT_DIR = str(TUTORIAL_DIR / "data" / "1_fif_input")
WORKING_DIR = str(TUTORIAL_DIR / "data" / "working")

# Derived paths (pipeline directory structure)
PREPROCESSED_FIF_DIR = str(Path(WORKING_DIR) / "1_fif_input")
PT_INPUT_DIR = str(Path(WORKING_DIR) / "2_pt_input")
PT_OUTPUT_DIR = str(Path(WORKING_DIR) / "3_pt_output")
FIF_OUTPUT_DIR = str(Path(WORKING_DIR) / "4_fif_output")

# =============================================================================
# OPTIONS
# =============================================================================

# TARGET_CHANNEL_COUNT = None   # no upsampling
# TARGET_CHANNEL_COUNT = 40     # upsample to N channels (greedy selection)
TARGET_CHANNEL_COUNT = ['AF3', 'AF4', 'F1', 'F2', 'FC1', 'FC2', 'CP1', 'CP2', 'PO3', 'PO4']
BAD_CHANNELS = ['Fz', 'Cz']    # Set to None to disable
GPU_DEVICE = 0

# =============================================================================
# STEP 1: Preprocessing (.fif → .pt)
# =============================================================================

if __name__ == "__main__":

    print("[1/4] Preprocessing...", flush=True)
    zuna_preprocessing(
        input_dir=INPUT_DIR,
        output_dir=PT_INPUT_DIR,
        save_preprocessed_fif=True,
        preprocessed_fif_dir=PREPROCESSED_FIF_DIR,
        target_channel_count=TARGET_CHANNEL_COUNT,
        bad_channels=BAD_CHANNELS,
    )

    # =============================================================================
    # STEP 2: Model Inference (.pt → .pt)
    # =============================================================================

    print("[2/4] Model inference...", flush=True)
    zuna_inference(
        input_dir=PT_INPUT_DIR,
        output_dir=PT_OUTPUT_DIR,
        gpu_device=GPU_DEVICE,
    )

    # =============================================================================
    # STEP 3: Reconstruction (.pt → .fif)
    # =============================================================================

    print("[3/4] Reconstructing FIF files...", flush=True)
    zuna_pt_to_fif(
        input_dir=PT_OUTPUT_DIR,
        output_dir=FIF_OUTPUT_DIR,
    )

    # =============================================================================
    # STEP 4: Visualization (optional)
    # =============================================================================

    print("[4/4] Visualizing pipeline outputs...", flush=True)
    zuna_plot(
        input_dir=INPUT_DIR,
        working_dir=WORKING_DIR,
        plot_pt=True,
        plot_fif=True,
    )

    print("Done. Output:", FIF_OUTPUT_DIR)
