#!/usr/bin/env python3
"""
Zuna pipeline — direct .fif path (EEGDataset_v4).

Reads .fif files directly, runs the model, and writes reconstructed .fif back out —
no .fif -> .pt -> .fif round trip (unlike run_zuna_pipeline.py).

    .fif  ->  EEGDataset_v4 (load, segment)  ->  model  ->  stitch chunks  ->  .fif  +  figures

Outputs under RECON_OUT (data/2_fif_output):
    full_reconstruction/<name>_raw.fif   model output everywhere
    hybrid/<name>_raw.fif                original input, model output ONLY on inferred cells
    hybrid/<name>_mask.npz               per (channel, sample) mask of what was inferred
Plus a full-duration, multi-channel overlay figure per file under FIGURES_DIR.

Weights load from a local checkpoint (CHECKPOINT); there is no HuggingFace download here.
Edit the constants below and run:  python run_zuna_pipeline_fif.py
"""

import os
import sys
import subprocess
from pathlib import Path

# ============================================================================= PATHS
TUTORIAL_DIR = Path(__file__).parent.resolve()
REPO_ROOT = TUTORIAL_DIR.parent

INPUT_DIR   = TUTORIAL_DIR / "data" / "1_fif_input"          # directory of input .fif files
RECON_OUT   = TUTORIAL_DIR / "data" / "2_fif_output"         # reconstructed .fif (full/ + hybrid/)
FIGURES_DIR = TUTORIAL_DIR / "data" / "working" / "FIGURES_FIF"
DUMP_DIR    = TUTORIAL_DIR / "data" / "working" / "v4_dump"  # eeg_eval logs/metrics

# Trained ZUNA checkpoint (local). config_infer_fif.yaml matches its architecture.
CHECKPOINT = "/data/groups/bci/checkpoints/bci/ZUNA2_5e-4_noALW2/checkpoints/0000548000"

# Vendored model app (zuna is a src-layout package, resolved from the repo tree).
APP_DIR     = REPO_ROOT / "src/zuna/inference/AY2l/lingua/apps/AY2latent_bci"
EEG_EVAL    = APP_DIR / "eeg_eval.py"
CONFIG_PATH = APP_DIR / "configs/config_infer_fif.yaml"
LINGUA_ROOT = APP_DIR.parent.parent                          # holds the `lingua` + `apps` packages

# ============================================================================= INFERENCE
GPU_DEVICE             = 1        # GPU id (check `nvidia-smi` for a free one), or "" for CPU
TOKENS_PER_BATCH       = 100000  # data.target_packed_seqlen
DATA_NORM              = 10.0    # rescale eeg to std ~= 0.1 (ZUNA expects this)
DIFFUSION_CFG          = 1.0     # 1.0 = no cfg
DIFFUSION_SAMPLE_STEPS = 50
PLOT_WINDOW_SEC        = 60      # seconds shown in the overlay figure (None = full recording)

# ============================================================================= WHAT GETS RECONSTRUCTED
# The model infers (reconstructs) exactly the "bad"/selected cells; everything else is kept.
# Sources of "bad" cells, combined:
#   1. BAD_* annotations in the .fif   -> those time spans (all channels)   [AUTOMATIC]
#   2. channels in the .fif info['bads'] -> whole channel                   [AUTOMATIC]
#   3. V4_DROP_CHANNELS                -> whole channel, by name            [manual, below]
#   4. V4_TARGET_CHANNEL_COUNT         -> zero-filled channels the model fills in (upsampling)
# For THIS example we rely on the input files' BAD_ annotations (1) + info['bads'] (2);
# no manual channel selection and no upsampling.
V4_SEGMENT_SEC          = 5.0            # segment window length (seconds)
V4_HIGHPASS_HZ          = 0.5            # highpass (Hz); None to skip
V4_MONTAGE              = "standard_1020"  # only used if a .fif lacks channel positions
V4_DROP_CHANNELS        = None           # e.g. ["Cz", "Pz"] to also repair these; None = off
V4_TARGET_CHANNEL_COUNT = None           # e.g. 40 to upsample+infer to N channels; None = off

# ============================================================================= RUN
def build_cmd():
    overrides = {
        "config":                     CONFIG_PATH,
        "dump_dir":                   DUMP_DIR.absolute(),
        "checkpoint.init_ckpt_path":  CHECKPOINT,
        "data.data_dir":              INPUT_DIR.absolute(),
        "data.target_packed_seqlen":  TOKENS_PER_BATCH,
        "data.data_norm":             DATA_NORM,
        "diffusion_cfg":              DIFFUSION_CFG,
        "diffusion_sample_steps":     DIFFUSION_SAMPLE_STEPS,
        "inference_figures_dir":      FIGURES_DIR,
        "data.v4_recon_save_fif":     "true",
        "data.v4_recon_out_dir":      RECON_OUT.absolute(),
        "data.v4_segment_sec":        V4_SEGMENT_SEC,
        "data.v4_montage":            V4_MONTAGE,
    }
    # Optional overrides — omitted when unset so the yaml default (`null`) stands.
    if V4_HIGHPASS_HZ is not None:
        overrides["data.v4_highpass_hz"] = V4_HIGHPASS_HZ
    if V4_TARGET_CHANNEL_COUNT is not None:
        overrides["data.v4_target_channel_count"] = V4_TARGET_CHANNEL_COUNT
    if V4_DROP_CHANNELS:
        overrides["data.v4_drop_channels"] = "[" + ",".join(V4_DROP_CHANNELS) + "]"
    return [sys.executable, str(EEG_EVAL)] + [f"{k}={v}" for k, v in overrides.items()]


def build_env():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_DEVICE)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(LINGUA_ROOT), str(APP_DIR), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    # This login node sits inside a SLURM allocation (SLURM_JOB_ID set) but without
    # SLURM_NTASKS, which sends lingua's distributed helpers down the SLURM path and
    # crashes. Drop SLURM_* so it runs as a plain single-GPU process (world_size=1).
    for k in [k for k in env if k.startswith("SLURM_")]:
        del env[k]
    return env


def make_overlay_figures():
    """Full-duration, multi-channel input-vs-reconstruction overlays (inferred cells shaded)."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from zuna.visualization.reconstruction_overlay import plot_reconstruction_overlay
    except Exception as e:
        print(f"  [fig] could not import plot_reconstruction_overlay ({e}); skipping figures.")
        return
    fig_dir = FIGURES_DIR / "reconstruction_overlays"
    for inp in sorted(INPUT_DIR.glob("*.fif")):
        # FifReconstructor writes "{base}_raw.fif" / "{base}_mask.npz" where base drops "_raw".
        # Derive the same base so inputs NOT named "*_raw.fif" (e.g. EPCTL02_n2.fif) still match.
        base = inp.stem.replace("_raw", "")
        mask_npz = RECON_OUT / "hybrid" / f"{base}_mask.npz"
        for kind in ("full_reconstruction", "hybrid"):
            recon = RECON_OUT / kind / f"{base}_raw.fif"
            if not recon.exists():
                print(f"  [fig] {kind}: no reconstruction for {inp.name}; skipping")
                continue
            out = fig_dir / f"{base}__{kind}.png"
            plot_reconstruction_overlay(
                input_fif=inp, recon_fif=recon, out_path=out,
                mask_npz=mask_npz, title=f"{base}  —  {kind}",
                window_sec=PLOT_WINDOW_SEC,
                # Preprocess the plotted "input" to the model's domain (resample-to-recon-rate is
                # automatic; apply the same highpass) so it's a fair comparison with the reconstruction.
                input_highpass_hz=V4_HIGHPASS_HZ,
            )
            print(f"  [fig] {kind}: {inp.name} -> {out}")


if __name__ == "__main__":
    # Run under the repo venv (its scipy/numpy are compatible; the system python is not).
    # If launched with a different interpreter, re-exec with venv/bin/python3.
    _venv_py = REPO_ROOT / "venv" / "bin" / "python3"
    if _venv_py.exists() and Path(sys.executable) != _venv_py:
        os.execv(str(_venv_py), [str(_venv_py), *sys.argv])

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"V4 config not found at {CONFIG_PATH}")
    if not EEG_EVAL.exists():
        raise FileNotFoundError(f"eeg_eval.py not found at {EEG_EVAL}")
    for d in (RECON_OUT, FIGURES_DIR, DUMP_DIR):
        d.mkdir(parents=True, exist_ok=True)

    cmd = build_cmd()
    print("[v4] direct-.fif inference")
    print(f"     input : {INPUT_DIR}")
    print(f"     recon : {RECON_OUT}")
    print(f"     figs  : {FIGURES_DIR}")
    print(f"     cmd   : {' '.join(cmd)}", flush=True)

    subprocess.run(cmd, env=build_env(), check=True)

    print("\n✓ inference + reconstruction complete.")
    print(f"  .fif: {RECON_OUT}/full_reconstruction/  and  {RECON_OUT}/hybrid/")
    make_overlay_figures()
    print(f"  figures: {FIGURES_DIR}/reconstruction_overlays/")
