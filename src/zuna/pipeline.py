"""
Zuna Complete Pipeline

This module provides functions to run the complete EEG reconstruction pipeline:
.fif → .pt → model inference → .pt → .fif

Each step can be run independently or as a complete pipeline.
"""

import os
from pathlib import Path
from collections import defaultdict

import mne


def inference(
    input_dir: str,
    output_dir: str,
    gpu_device: int|str = 0, 
    tokens_per_batch: int|None = None,
    data_norm: float|None = None,
    diffusion_cfg: float = 1.0,
    diffusion_sample_steps: int = 50,
    plot_eeg_signal_samples: bool = False,
    inference_figures_dir: str = "./inference_figures",
) -> None:
    """
    Run Zuna model inference on .pt files.
    Model weights are automatically downloaded from HuggingFace.

    Args:
        input_dir: Directory containing preprocessed .pt files
        output_dir: Directory to save model output .pt files
        gpu_device: GPU device ID (default: 0), or "" for CPU
        tokens_per_batch: Number of tokens per batch - Increase this number for higher GPU utilization.
        data_norm: Data normalization factor denominator to rescale eeg data to have std = 0.1
                   NOTE: ZUNA was trained on and expects eeg data to have std = 0.1
        diffusion_cfg: Diffusion process in .sample - Default is 1.0 (i.e., no cfg)
        diffusion_sample_steps: Number of steps in the diffusion process - Default is 50
        plot_eeg_signal_samples: Plot raw eeg for data and model reconstruction for single samples inside inference code.
                                NOTE: Will use GPU very inefficiently if True. Set to False when running at scale
        inference_figures_dir: Directory to save inference figures    
    """
    import subprocess

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load the base config file
    config_path = Path(__file__).parent / "inference/AY2l/lingua/apps/AY2latent_bci/configs/config_infer.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Build command to run eeg_eval.py
    eeg_eval_script = Path(__file__).parent / "inference/AY2l/lingua/apps/AY2latent_bci/eeg_eval.py"

    # Build up command to run eeg_eval.py
    cmd = [
        "python3",
        str(eeg_eval_script),
        f"config={config_path}",
        f"data.data_dir={str(Path(input_dir).absolute())}",
        f"data.export_dir={str(output_path.absolute())}",
        f"diffusion_cfg={diffusion_cfg}",
        f"diffusion_sample_steps={diffusion_sample_steps}",
        f"plot_eeg_signal_samples={plot_eeg_signal_samples}",
        f"inference_figures_dir={inference_figures_dir}",
    ]

    # Add optional parameters
    if tokens_per_batch is not None:
        cmd.append(f"data.target_packed_seqlen={tokens_per_batch}")
    if data_norm is not None:
        cmd.append(f"data.data_norm={data_norm}")

    # Set environment variable for GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_device)

    # Run the command
    result = subprocess.run(cmd, env=env, check=True)

    print(f"✓ Inference complete")


# Masks are stored at TOKEN resolution: one column per NUM_FINE_TIME_PTS samples. The model only
# ever drops/keeps whole coarse-time tokens (tf = num_fine_time_pts samples ≈ 0.125 s at 256 Hz), so
# a per-sample mask carries tf× more columns than it can act on. Keep this in sync with the model's
# `num_fine_time_pts` (config_infer_fif.yaml). n_times + sfreq travel in the .npz so it round-trips.
NUM_FINE_TIME_PTS = 32


def mask_samples_to_tokens(mask_samples, tf: int = NUM_FINE_TIME_PTS):
    """(C, N) per-sample bool -> (C, ceil(N/tf)) per-token bool; a token is bad if ANY of its samples is."""
    import numpy as np
    m = np.asarray(mask_samples, dtype=bool)
    C, N = m.shape
    n_tok = (N + tf - 1) // tf
    pad = n_tok * tf - N
    if pad:
        m = np.concatenate([m, np.zeros((C, pad), dtype=bool)], axis=1)
    return m.reshape(C, n_tok, tf).any(axis=2)


def mask_tokens_to_samples(mask_tokens, tf: int, n_times: int):
    """(C, n_tok) per-token bool -> (C, n_times) per-sample bool by repeating each token tf times."""
    import numpy as np
    return np.repeat(np.asarray(mask_tokens, dtype=bool), tf, axis=1)[:, :n_times]


def load_mask_npz_as_samples(npz_path, target_ch_names, n_times: int):
    """Read any mask .npz (token- or sample-resolution) -> (len(target_ch_names), n_times) bool, aligned
    by channel name (case-insensitive). Token resolution is detected via the stored num_fine_time_pts."""
    import numpy as np
    z = np.load(str(npz_path), allow_pickle=True)
    m = np.asarray(z["mask"]).astype(bool)
    names = [str(x) for x in z["ch_names"]]
    tf = int(z["num_fine_time_pts"]) if "num_fine_time_pts" in z.files else None
    W = m.shape[1]
    if tf is not None and W == (n_times + tf - 1) // tf and W != n_times:
        idx = np.minimum(np.arange(n_times) // tf, W - 1)          # token-resolution
    elif W == n_times:
        idx = np.arange(n_times)                                   # sample-resolution
    else:
        idx = np.minimum((np.arange(n_times) * (W / n_times)).astype(int), W - 1)  # other duration
    lower = {n.lower(): i for i, n in enumerate(names)}
    out = np.zeros((len(target_ch_names), n_times), dtype=bool)
    for i, c in enumerate(target_ch_names):
        if c.lower() in lower:
            out[i] = m[lower[c.lower()]][idx]
    return out


def write_bad_mask(
    fif_path: str,
    out_npz: str,
    bad_channels=None,
    bad_segments=None,
    base_mask_npz: str | None = None,
    num_fine_time_pts: int = NUM_FINE_TIME_PTS,
) -> str:
    """Build a per-file bad-mask .npz (TOKEN resolution) for the v4 reconstruction path.

    The mask marks the cells the model should reconstruct. Combine any of:
      - bad_channels: list of channel names -> whole channel marked bad.
      - bad_segments: list of tuples, times in DATA-RELATIVE seconds:
            (start, stop)          -> mark that span on ALL channels
            (start, stop, "Cz")    -> mark that span on channel "Cz" only
      - base_mask_npz: an existing mask .npz to start from (e.g. a UI-generated mask), token- or
        sample-resolution; the above are UNIONed on top of it.
    Output format (same as the reconstructor's output masks): npz with
      mask (n_ch, n_tokens) bool, ch_names (array of str), sfreq (float),
      num_fine_time_pts (int, samples per token), n_times (int, original sample count).
    """
    import numpy as np

    raw = mne.io.read_raw_fif(str(fif_path), preload=False, verbose="ERROR")
    ch_names = list(raw.ch_names)
    sfreq = float(raw.info["sfreq"])
    N = raw.n_times
    row = {c.lower(): i for i, c in enumerate(ch_names)}   # case-insensitive channel lookup

    # Build at per-sample resolution (segment times are sample-accurate), then collapse to tokens.
    mask = np.zeros((len(ch_names), N), dtype=bool)
    if base_mask_npz is not None and Path(base_mask_npz).exists():
        mask |= load_mask_npz_as_samples(base_mask_npz, ch_names, N)

    for c in (bad_channels or []):
        if c.lower() in row:
            mask[row[c.lower()], :] = True

    for seg in (bad_segments or []):
        start, stop = float(seg[0]), float(seg[1])
        s = max(0, int(round(start * sfreq)))
        e = min(N, int(round(stop * sfreq)))
        if e <= s:
            continue
        if len(seg) >= 3 and seg[2] is not None:      # per-channel segment
            if seg[2].lower() in row:
                mask[row[seg[2].lower()], s:e] = True
        else:                                          # all channels
            mask[:, s:e] = True

    tf = int(num_fine_time_pts)
    mask_tok = mask_samples_to_tokens(mask, tf)
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_npz), mask=mask_tok, ch_names=np.array(ch_names),
                        sfreq=np.float32(sfreq), num_fine_time_pts=np.int64(tf), n_times=np.int64(N))
    return out_npz


def reconstruct_fif(
    input_dir: str,
    output_dir: str,
    figures_dir: str,
    tmp_dir: str | None = None,
    gpu_device: int | str = 0,
    segment_sec: float = 5.0,
    highpass_hz: float | None = 0.5,
    montage: str = "standard_1020",
    repair_channels=None,
    target_channel_count=None,
    mask_dir: str | None = None,
    bad_segments=None,
    use_fif_annotations: bool = True,
    annotate_infill: bool = True,
    save_preprocessed: bool = False,
    bad_token_overlap: float = 0.0,
    window_sec: float | None = 60,
    diffusion_cfg: float = 1.0,
    sample_steps: int = 50,
) -> None:
    """Reconstruct .fif files directly with the v4 model (EEGDataset_v4), no .pt round-trip.

    Reads each .fif in `input_dir`, reconstructs the "bad"/selected cells, and writes:
      <output_dir>/full_reconstruction/<base>_raw.fif   model output everywhere
      <output_dir>/hybrid/<base>_raw.fif                original, model only on inferred cells
    plus a full-duration overlay per file in `figures_dir`. The inferred (channel, time) cells are
    recorded as ZUNA1.1_infilled annotations on both output .fif files (no separate mask .npz). When `save_preprocessed` is True, the
    fully-preprocessed input (resampled + filtered + montage + info['bads'] + upsampled channels,
    exactly what the model ingests) is also written to <output_dir>/fif_input_preprocessed/<base>_raw.fif.

    The reconstruction mask is the UNION of: the .fif's BAD_ annotations + info['bads'] (automatic),
    `repair_channels` (names), `target_channel_count` (upsampled channels, placed far from existing
    electrodes), `bad_segments` (manual (start, stop[, channel]) tuples), and any per-file masks in
    `mask_dir` (e.g. from a UI). Model weights load from HuggingFace (see eeg_eval Load_from_HF).

    BAD_ annotations may be channel-specific: an MNE annotation with a populated `ch_names` marks
    that time span bad only on those channels; an empty `ch_names` (the default) marks all channels.
    Annotation onsets are read on MNE's clock: when the file has an orig_time / first_samp, the
    absolute onset is converted back to a data-relative sample (first_samp subtracted).

    `bad_token_overlap` controls how a segment maps onto the model's coarse tokens (num_fine_time_pts
    samples each): 0.0 (default) marks a token bad on ANY overlap (spans widen out to whole tokens);
    0.5 marks a token only when the majority of it is bad (tight edges round inward, so the infilled
    region hugs the annotation instead of over-covering by up to one token per side).

    When `annotate_infill` is True (default), the output .fif files carry 'ZUNA1.1_infilled'
    annotations recording exactly which (channel, time) cells were reconstructed — per-channel where
    the infill was channel-specific, global where every channel was infilled. A channel infilled for
    its whole duration gets a full-duration annotation and is dropped from the output info['bads'].
    """
    import sys
    import subprocess

    app_dir = Path(__file__).parent / "inference/AY2l/lingua/apps/AY2latent_bci"
    eeg_eval = app_dir / "eeg_eval.py"
    config = app_dir / "configs/config_infer_fif.yaml"
    lingua_root = app_dir.parent.parent
    for p in (config, eeg_eval):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    input_dir = Path(input_dir); output_dir = Path(output_dir); figures_dir = Path(figures_dir)
    tmp_dir = Path(tmp_dir) if tmp_dir is not None else output_dir.parent / "tmp"
    for d in (output_dir, figures_dir, tmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Effective mask dir = UI masks and/or manual bad_segments, UNIONed per file. Built into tmp.
    eff_mask_dir = None
    if bad_segments:
        eff_mask_dir = tmp_dir / "input_masks"
        for fif in sorted(input_dir.glob("*.fif")):
            base = fif.stem.replace("_raw", "")
            ui = (Path(mask_dir) / f"{base}_mask.npz") if mask_dir else None
            write_bad_mask(str(fif), str(eff_mask_dir / f"{base}_mask.npz"),
                           bad_segments=bad_segments,
                           base_mask_npz=str(ui) if (ui and ui.exists()) else None)
    elif mask_dir:
        eff_mask_dir = Path(mask_dir)

    overrides = {
        "config": config,
        "dump_dir": tmp_dir.absolute(),
        "inference_figures_dir": tmp_dir.absolute(),   # eeg_eval internal output -> tmp
        "data.data_dir": input_dir.absolute(),
        "data.v4_recon_save_fif": "true",
        "data.v4_recon_out_dir": output_dir.absolute(),
        "data.v4_segment_sec": segment_sec,
        "data.v4_montage": montage,
        "data.v4_use_fif_annotations": "true" if use_fif_annotations else "false",
        "data.v4_recon_annotate_infill": "true" if annotate_infill else "false",
        "data.v4_recon_save_preprocessed": "true" if save_preprocessed else "false",
        "data.v4_bad_token_overlap": bad_token_overlap,
        "diffusion_cfg": diffusion_cfg,
        "diffusion_sample_steps": sample_steps,
    }
    if highpass_hz is not None:
        overrides["data.v4_highpass_hz"] = highpass_hz
    if repair_channels:
        overrides["data.v4_drop_channels"] = "[" + ",".join(repair_channels) + "]"
    if target_channel_count is not None:
        # int N  -> auto-add channels spread far from existing electrodes;
        # list of names -> add exactly those channels at their montage positions.
        if isinstance(target_channel_count, (list, tuple)):
            overrides["data.v4_target_channel_count"] = "[" + ",".join(map(str, target_channel_count)) + "]"
        else:
            overrides["data.v4_target_channel_count"] = target_channel_count
    if eff_mask_dir is not None:
        overrides["data.v4_mask_dir"] = Path(eff_mask_dir).absolute()
    cmd = [sys.executable, str(eeg_eval)] + [f"{k}={v}" for k, v in overrides.items()]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    env["WANDB_MODE"] = "disabled"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(lingua_root), str(app_dir), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    # Run as a plain single-GPU process: this login node is inside a SLURM allocation without
    # SLURM_NTASKS, which sends lingua's distributed helpers down the SLURM path and crashes.
    for k in [k for k in env if k.startswith("SLURM_")]:
        del env[k]

    print(f"[v4] direct-.fif reconstruction: {input_dir} -> {output_dir}", flush=True)
    subprocess.run(cmd, env=env, check=True)

    # Full-duration input-vs-reconstruction overlays (input preprocessed to the model's domain).
    from .visualization.reconstruction_overlay import plot_reconstruction_overlay
    for fif in sorted(input_dir.glob("*.fif")):
        base = fif.stem.replace("_raw", "")
        mask_npz = output_dir / "hybrid" / f"{base}_mask.npz"
        for kind in ("full_reconstruction", "hybrid"):
            recon = output_dir / kind / f"{base}_raw.fif"
            if not recon.exists():
                print(f"  [fig] {kind}: no reconstruction for {fif.name}; skipping")
                continue
            plot_reconstruction_overlay(
                input_fif=fif, recon_fif=recon, out_path=figures_dir / f"{base}__{kind}.png",
                mask_npz=mask_npz, title=f"{base}  —  {kind}",
                window_sec=window_sec, input_highpass_hz=highpass_hz,
            )
    print(f"✓ reconstruction complete: {output_dir} ; figures: {figures_dir}")


def pt_to_fif(
    input_dir: str,
    output_dir: str,
) -> None:
    """
    Convert model output .pt files back to .fif (MNE Raw) format.

    Reads .pt files produced by model inference, reverses the normalization
    and epoching applied during preprocessing, and reconstructs continuous
    .fif files. Each .pt file contains metadata (channel names, sampling
    frequency, normalization parameters) needed for reconstruction.

    Multiple .pt files that originated from the same source .fif file are
    automatically detected (via metadata) and stitched back together into
    a single continuous recording.

    The reconstructed .fif files will have:
        - The same channel names and montage as the preprocessed input
        - Signal values denormalized back to original scale (microvolts)
        - Epochs concatenated back into a continuous recording

    Args:
        input_dir: Directory containing .pt files from model inference.
            Each .pt file must contain 'data_reconstructed', 'metadata'
            (with channel names, sfreq, normalization params), and
            'channel_positions' keys.
        output_dir: Directory to save reconstructed .fif files.

    Example:
        >>> from zuna import pt_to_fif
        >>> pt_to_fif(
        ...     input_dir="/data/eeg/working/3_pt_output",
        ...     output_dir="/data/eeg/working/4_fif_output",
        ... )
    """
    from .preprocessing.io import load_pt, pt_to_raw
    from .preprocessing.normalizer import Normalizer
    import torch
    import numpy as np

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all PT files
    input_path = Path(input_dir)
    pt_files = list(input_path.glob("*.pt"))

    if len(pt_files) == 0:
        print("Reconstruction: no .pt files found.")
        return

    # Group PT files by original source filename
    source_groups = defaultdict(list)
    for pt_file in pt_files:
        try:
            pt_data = load_pt(str(pt_file))
            metadata = pt_data.get('metadata', {})
            original_filename = metadata.get('original_filename', pt_file.name)
            source_groups[original_filename].append(pt_file)
        except Exception:
            source_groups[pt_file.name].append(pt_file)

    # Convert each group
    successful = 0
    failed = 0

    for original_filename, pt_file_group in source_groups.items():
        try:
            pt_file_group = sorted(pt_file_group)

            # Check input_type from first file's metadata
            first_pt = load_pt(str(pt_file_group[0]))
            first_metadata = first_pt.get('metadata', {})
            is_epochs_input = first_metadata.get('input_type') == 'epochs'

            if is_epochs_input:
                # Epoch path: reconstruct as mne.Epochs, save as *_epo.fif
                all_epoch_data = []
                channel_names = first_metadata.get('channel_names', [])
                sfreq = first_metadata.get('resampled_sfreq', 256.0)
                positions = None

                for pt_file in pt_file_group:
                    pt_data = load_pt(str(pt_file))
                    metadata = pt_data.get('metadata', {})

                    for tensor in pt_data['data']:
                        if tensor is None:
                            continue
                        epoch = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
                        all_epoch_data.append(epoch)

                    if positions is None and len(pt_data['channel_positions']) > 0:
                        pos = pt_data['channel_positions'][0]
                        positions = pos.numpy() if isinstance(pos, torch.Tensor) else pos

                if len(all_epoch_data) == 0:
                    raise ValueError("No valid epochs found")

                # Denormalize
                epoch_array = np.stack(all_epoch_data)
                if 'reversibility' in first_metadata:
                    epoch_array = Normalizer.denormalize(epoch_array, first_metadata['reversibility'])

                # Create MNE Epochs
                n_channels = epoch_array.shape[1]
                if len(channel_names) > n_channels:
                    channel_names = channel_names[:n_channels]
                elif len(channel_names) < n_channels:
                    channel_names = channel_names + [f'Ch{i+1}' for i in range(len(channel_names), n_channels)]

                info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
                n_epochs = epoch_array.shape[0]
                n_samples = epoch_array.shape[2]
                events = np.column_stack([
                    np.arange(n_epochs) * n_samples,  # sample onset
                    np.zeros(n_epochs, dtype=int),     # previous event
                    np.ones(n_epochs, dtype=int),       # event id
                ])
                epochs_obj = mne.EpochsArray(epoch_array, info, events=events, verbose=False)

                # Set montage from positions
                if positions is not None and positions.shape[0] == len(channel_names):
                    ch_pos = {ch: pos for ch, pos in zip(channel_names, positions)}
                    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
                    epochs_obj.set_montage(montage, verbose=False)

                # Save as epoch FIF
                base_name = original_filename.replace('_epo.fif', '').replace('-epo.fif', '').replace('.fif', '')
                epo_path = output_path / (base_name + "_epo.fif")
                epochs_obj.save(str(epo_path), overwrite=True, verbose=False)
                successful += 1

            else:
                # Raw path: original behavior — concatenate into continuous Raw
                raw_objects = [pt_to_raw(str(f)) for f in pt_file_group]
                if len(raw_objects) > 1:
                    combined_raw = mne.concatenate_raws(raw_objects, preload=True)
                else:
                    combined_raw = raw_objects[0]

                base_name = original_filename.replace('.fif', '').replace('.FIF', '')
                fif_path = output_path / (base_name + ".fif")
                combined_raw.save(str(fif_path), overwrite=True)
                successful += 1

        except Exception as e:
            failed += 1
            print(f"  Error: {original_filename}: {e}")

    print(f"Reconstruction: {successful}/{successful + failed} files converted.")

