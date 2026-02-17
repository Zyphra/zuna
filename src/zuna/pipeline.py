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
    Run model inference on .pt files.
    Zuna model weights are automatically downloaded from HuggingFace.

    Args:
        input_dir: Directory containing preprocessed .pt files
        output_dir: Directory to save model output .pt files
        gpu_device: GPU device ID (default: 0), or "" for CPU
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

            # Convert all PT files to Raw objects and concatenate
            raw_objects = [pt_to_raw(str(f)) for f in pt_file_group]
            if len(raw_objects) > 1:
                combined_raw = mne.concatenate_raws(raw_objects, preload=True)
            else:
                combined_raw = raw_objects[0]

            # Save as FIF
            base_name = original_filename.replace('.fif', '').replace('.FIF', '')
            fif_path = output_path / (base_name + ".fif")
            combined_raw.save(str(fif_path), overwrite=True)
            successful += 1

        except Exception as e:
            failed += 1
            print(f"  Error: {original_filename}: {e}")

    print(f"Reconstruction: {successful}/{successful + failed} files converted.")

