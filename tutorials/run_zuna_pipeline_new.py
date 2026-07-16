#!/usr/bin/env python3
"""
Zuna pipeline — direct .fif reconstruction (the standard model).

Reads .fif files, reconstructs the "bad"/selected cells with the model, and writes .fif back out
(no .fif -> .pt -> .fif round-trip). Edit the constants below and run:

    python run_zuna_pipeline_fif.py

Outputs under OUTPUT_DIR:
    full_reconstruction/<name>_raw.fif   model output everywhere
    hybrid/<name>_raw.fif                original input, model output ONLY on the inferred cells
    hybrid/<name>_mask.npz               per (channel, sample) mask of what was inferred
plus a full-duration input-vs-reconstruction overlay per file in FIGURES_DIR.
"""

import os
import sys
from pathlib import Path

# Run under the repo venv (compatible scipy/numpy) + make the `zuna` package importable.
_REPO = Path(__file__).resolve().parent.parent
_VENV_PY = _REPO / "venv" / "bin" / "python3"
if _VENV_PY.exists() and Path(sys.executable) != _VENV_PY:
    os.execv(str(_VENV_PY), [str(_VENV_PY), *sys.argv])
sys.path.insert(0, str(_REPO / "src"))

from zuna import reconstruct_fif

# ============================================================================= PATHS
DATA        = Path(__file__).parent / "data"
INPUT_DIR   = DATA / "1_fif_input"     # input .fif files
OUTPUT_DIR  = DATA / "2_fif_output"    # reconstructed .fif (full_reconstruction/ + hybrid/)
FIGURES_DIR = DATA / "figures"         # comparison figures
TMP_DIR     = DATA / "tmp"             # logs/metrics (throwaway)

# ============================================================================= OPTIONS
GPU_DEVICE  = 1            # GPU id (check `nvidia-smi`), or "" for CPU
SEGMENT_SEC = 5.0          # model segment window length (seconds)
HIGHPASS_HZ = 0.5          # highpass applied to each .fif before the model (None to skip)
MONTAGE     = "standard_1020"   # used only to add positions when a .fif lacks them
WINDOW_SEC  = 60           # seconds shown in each overlay figure (None = full recording)

# ============================================================================= WHAT GETS RECONSTRUCTED
IMPORT_FIF_BAD_SEGMENTS = True                # use the .fif's own BAD_ annotations (whole-channel bads always used)
REPAIR_CHANNELS         = ["Cz"]                # existing channel(s) to reconstruct completely
ADD_CHANNELS            = ["Fz", "Pz"]          # NEW channels to add + infer at their montage location
TARGET_CHANNEL_COUNT    = None                  # OR auto-add up to N total channels (far apart), e.g. 40
BAD_SEGMENTS            = [(5, 6),              # 5-6 s bad on ALL channels
                           (10, 11, "C3"),       # 10-11 s bad on C3 only
                           (10, 11, "C4")]       # 10-11 s bad on C4 only   (-> "just for some" channels)
MASK_DIR                = None                  # dir of per-file "<base>_mask.npz" (channel x sample bool)

# ============================================================================= RUN
if __name__ == "__main__":
    reconstruct_fif(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        figures_dir=FIGURES_DIR,
        tmp_dir=TMP_DIR,
        gpu_device=GPU_DEVICE,
        segment_sec=SEGMENT_SEC,
        highpass_hz=HIGHPASS_HZ,
        montage=MONTAGE,
        window_sec=WINDOW_SEC,
        repair_channels=REPAIR_CHANNELS,
        # ADD_CHANNELS (specific names) and TARGET_CHANNEL_COUNT (auto N) drive the same upsampling;
        # a name-list adds those exact channels, an int auto-adds far-apart ones.
        target_channel_count=(ADD_CHANNELS if ADD_CHANNELS else TARGET_CHANNEL_COUNT),
        bad_segments=BAD_SEGMENTS,
        mask_dir=MASK_DIR,
        use_fif_annotations=IMPORT_FIF_BAD_SEGMENTS,
    )
