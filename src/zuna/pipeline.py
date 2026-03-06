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

