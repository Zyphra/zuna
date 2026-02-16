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

"""
INSTRUCTIONS: (for ReadMe etc)

* pip install zuna
* prepare data: have it in .fif format, with a channel montage applied 
* then go to tutorials/getting_started.py


"""

from pathlib import Path
from zuna.pipeline import run_zuna_pipeline
import zuna

# =============================================================================
# LOOK AT THE HELP DOCUMENTATION FOR MORE INFORMATION 
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

TUTORIAL_DIR = Path(__file__).parent.resolve()
INPUT_DIR = str(TUTORIAL_DIR / "data" / "1_fif_input")
WORKING_DIR = str(TUTORIAL_DIR / "data" / "working")

# =============================================================================
# OPTIONS
# =============================================================================

# TARGET_CHANNEL_COUNT = None   # no upsampling
# TARGET_CHANNEL_COUNT = 40     # upsample to N channels (greedy selection)
TARGET_CHANNEL_COUNT = ['AF3', 'AF4', 'F1', 'F2', 'FC1', 'FC2', 'CP1', 'CP2', 'PO3', 'PO4'] # add specific channels from 10-05 montage
BAD_CHANNELS = ['Fz', 'Cz']  # List of channels to zero out for interpolation testing, Set to None to disable: BAD_CHANNELS = None
KEEP_INTERMEDIATE_FILES = True  # If False, deletes .pt files after reconstruction
GPU_DEVICE = 0                  # An integer pointing at which GPU to use. Or, if GPU_DEVICE is set to "", run pipeline on the CPU.
PLOT_PT_COMPARISON = True       # Plot .pt file comparisons (preprocessed vs model output)
PLOT_FIF_COMPARISON = True      # Plot .fif file comparisons (preprocessed vs reconstructed)

# =============================================================================
# RUN PIPELINE
# =============================================================================

if __name__ == "__main__":
    run_zuna_pipeline(
        input_dir=INPUT_DIR,
        working_dir=WORKING_DIR,
        target_channel_count=TARGET_CHANNEL_COUNT,
        bad_channels=BAD_CHANNELS,
        keep_intermediate_files=KEEP_INTERMEDIATE_FILES,
        gpu_device=GPU_DEVICE,
        plot_pt_comparison=PLOT_PT_COMPARISON,
        plot_fif_comparison=PLOT_FIF_COMPARISON,
    )
