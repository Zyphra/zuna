"""
Zuna Complete Pipeline

This module provides functions to run the complete EEG reconstruction pipeline:
.fif → .pt → model inference → .pt → .fif

Each step can be run independently or as a complete pipeline.
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Optional, Union, List

import mne


def zuna_preprocessing(
    input_dir: str,
    output_dir: str,
    target_sfreq: float = 256.0,
    epoch_duration: float = 5.0,
    apply_notch_filter: bool = False,
    apply_highpass_filter: bool = True,
    apply_average_reference: bool = True,
    save_preprocessed_fif: bool = True,
    preprocessed_fif_dir: Optional[str] = None,
    drop_bad_channels: bool = False,
    drop_bad_epochs: bool = False,
    zero_out_artifacts: bool = False,
    target_channel_count: Optional[Union[int, List[str]]] = None,
    bad_channels: Optional[List[str]] = None,
) -> None:
    """
    Preprocess .fif files to .pt format.

    Reads raw EEG .fif files, applies filtering, resampling, epoching, and
    normalization, then saves the result as .pt (PyTorch) files ready for
    model inference. Optionally saves a preprocessed .fif copy for later
    comparison.

    Processing steps:
        1. Load raw .fif file
        2. Resample to target_sfreq (default 256 Hz)
        3. Apply highpass filter at 0.5 Hz (optional)
        4. Apply notch filter for line noise removal (optional)
        5. Apply average reference (optional)
        6. Zero out bad channels if specified
        7. Epoch into fixed-length segments (default 5 seconds)
        8. Normalize signal per epoch
        9. Upsample/add channels if target_channel_count is set
        10. Save as .pt files

    Args:
        input_dir: Directory containing input .fif files.
        output_dir: Directory to save preprocessed .pt files.
        target_sfreq: Target sampling frequency in Hz (default: 256.0).
        epoch_duration: Duration of each epoch in seconds (default: 5.0).
        apply_notch_filter: Apply automatic notch filter to remove line noise
            at detected frequencies (default: False).
        apply_highpass_filter: Apply 0.5 Hz highpass filter (default: True).
        apply_average_reference: Apply average reference (default: True).
        save_preprocessed_fif: Save a preprocessed .fif file alongside the .pt
            output, useful for comparing input vs output (default: True).
        preprocessed_fif_dir: Directory for preprocessed .fif files. If None,
            defaults to input_dir/preprocessed.
        drop_bad_channels: Automatically detect and remove bad channels
            (default: False). Not recommended for most use cases.
        drop_bad_epochs: Automatically detect and remove bad epochs
            (default: False). Not recommended for most use cases.
        zero_out_artifacts: Zero out artifact samples instead of dropping
            them (default: False).
        target_channel_count: Controls channel upsampling/selection.
            - None: keep original channels, no upsampling (default).
            - int (e.g. 40): greedy selection to N channels from 10-05 montage.
            - list of str (e.g. ['Cz', 'Pz']): add these specific channels
              from the 10-05 montage via spherical spline interpolation.
        bad_channels: List of channel names to zero out (e.g. ['Cz', 'Fz']).
            These channels remain in the data but their values are set to zero.
            Useful for testing interpolation. Set to None to disable (default: None).

    Example:
        >>> from zuna.pipeline import zuna_preprocessing
        >>> zuna_preprocessing(
        ...     input_dir="/data/eeg/raw_fif",
        ...     output_dir="/data/eeg/pt_files",
        ...     target_channel_count=['AF3', 'AF4', 'F1', 'F2'],
        ...     bad_channels=['Cz'],
        ... )
    """
    from .preprocessing.batch import process_directory

    # Setup preprocessed FIF directory if not specified
    if save_preprocessed_fif and preprocessed_fif_dir is None:
        preprocessed_fif_dir = str(Path(input_dir) / "preprocessed")

    process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        target_sfreq=target_sfreq,
        epoch_duration=epoch_duration,
        apply_notch_filter=apply_notch_filter,
        apply_highpass_filter=apply_highpass_filter,
        apply_average_reference=apply_average_reference,
        save_preprocessed_fif=save_preprocessed_fif,
        preprocessed_fif_dir=preprocessed_fif_dir,
        drop_bad_channels=drop_bad_channels,
        drop_bad_epochs=drop_bad_epochs,
        zero_out_artifacts=zero_out_artifacts,
        target_channel_count=target_channel_count,
        bad_channels=bad_channels,
    )


def zuna_inference(
    input_dir: str,
    output_dir: str,
    gpu_device: int = 0
) -> None:
    """
    Run model inference on .pt files.

    Model weights are automatically downloaded from HuggingFace.

    Args:
        input_dir: Directory containing preprocessed .pt files
        output_dir: Directory to save model output .pt files
        gpu_device: GPU device ID (default: 0)
    """
    from omegaconf import OmegaConf
    import subprocess

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load the base config file
    config_path = Path(__file__).parent / "inference/AY2l/lingua/apps/AY2latent_bci/configs/config_bci_eval.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Load and modify config with our paths
    config = OmegaConf.load(str(config_path))
    config.data.data_dir = str(Path(input_dir).absolute())
    config.data.export_dir = str(output_path.absolute())
    config.dump_dir = str(output_path.absolute())

    # Save modified config to temporary file
    temp_config_path = output_path / "temp_config.yaml"
    OmegaConf.save(config, str(temp_config_path))

    # Build command to run eeg_eval.py
    eeg_eval_script = Path(__file__).parent / "inference/AY2l/lingua/apps/AY2latent_bci/eeg_eval.py"

    cmd = [
        "python3",
        str(eeg_eval_script),
        f"config={temp_config_path}"
    ]

    # Set environment variable for GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_device)

    # Run the command
    result = subprocess.run(cmd, env=env, check=True)

    # Clean up temporary files created during inference
    temp_config_path.unlink()

    # Remove other temporary files/folders created by eeg_eval.py
    cleanup_files = [
        output_path / "checkpoints",  # folder
        output_path / "config.yaml",
        output_path / "metrics.jsonl",
        output_path / "train.log"
    ]

    for path in cleanup_files:
        try:
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        except Exception:
            pass  # Ignore cleanup errors

    print(f"✓ Inference complete")


def zuna_pt_to_fif(
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
        >>> from zuna.pipeline import zuna_pt_to_fif
        >>> zuna_pt_to_fif(
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


def zuna_plot(
    input_dir: str,
    working_dir: str,
    plot_pt: bool = False,
    plot_fif: bool = True,
) -> None:
    """
    Generate comparison plots between pipeline input and output.

    Compares preprocessed vs reconstructed files to visually inspect
    model quality. Plots are saved as images to working_dir/FIGURES/.

    Expects the standard pipeline directory structure under working_dir:
        1_fif_input/   - Preprocessed .fif files
        2_pt_input/    - Preprocessed .pt files
        3_pt_output/   - Model output .pt files
        4_fif_output/  - Reconstructed .fif files

    Args:
        input_dir: Directory containing the original input .fif files.
        working_dir: Working directory containing pipeline outputs.
        plot_pt: Compare .pt files (preprocessed vs model output).
            Shows per-epoch signal comparisons (default: False).
        plot_fif: Compare .fif files (preprocessed vs reconstructed).
            Shows full-recording signal overlays (default: True).

    Example:
        >>> from zuna.pipeline import zuna_plot
        >>> zuna_plot(
        ...     input_dir="/data/eeg/raw_fif",
        ...     working_dir="/data/eeg/working",
        ...     plot_fif=True,
        ... )
    """
    from .visualization.compare import compare_pipeline

    working_path = Path(working_dir)
    figures_dir = working_path / "FIGURES"
    figures_dir.mkdir(parents=True, exist_ok=True)

    compare_pipeline(
        input_dir=input_dir,
        fif_input_dir=str(working_path / "1_fif_input"),
        fif_output_dir=str(working_path / "4_fif_output"),
        pt_input_dir=str(working_path / "2_pt_input"),
        pt_output_dir=str(working_path / "3_pt_output"),
        output_dir=str(figures_dir),
        plot_pt=plot_pt,
        plot_fif=plot_fif,
    )


def run_zuna_pipeline(
    input_dir: str,
    working_dir: str,
    target_channel_count: Optional[Union[int, List[str]]] = None,
    bad_channels: Optional[List[str]] = None,
    keep_intermediate_files: bool = True,
    gpu_device: int = 0,
    plot_pt_comparison: bool = False,
    plot_fif_comparison: bool = False,
):
    """
    Run the complete Zuna pipeline: .fif → .pt → model → .pt → .fif

    This is the main entry point for running the full pipeline. It chains
    together preprocessing, model inference, and reconstruction, managing
    all intermediate directories automatically.

    The pipeline creates this directory structure under working_dir:
        working_dir/
            1_fif_input/      - Preprocessed .fif files (for comparison)
            2_pt_input/       - Preprocessed .pt files (model input)
            3_pt_output/      - Model output .pt files
            4_fif_output/     - Final reconstructed .fif files

    Args:
        input_dir: Directory containing input .fif files. Files must have
            a channel montage set (e.g. standard 10-20).
        working_dir: Working directory where all intermediate and output
            files will be stored in subdirectories.
        target_channel_count: Controls channel upsampling/selection.
            - None: keep original channels, no upsampling (default).
            - int (e.g. 40): greedy selection to N channels from 10-05 montage.
            - list of str (e.g. ['Cz', 'Pz']): add these specific channels
              from the 10-05 montage via spherical spline interpolation.
        bad_channels: List of channel names to zero out (e.g. ['Cz', 'Fz']).
            Channels remain in the data but values are set to zero.
            Set to None to disable (default: None).
        keep_intermediate_files: Keep .pt files after reconstruction. Set to
            False to save disk space (default: True).
        gpu_device: GPU device ID for model inference (default: 0).
        plot_pt_comparison: Generate plots comparing input vs output .pt
            files (default: False).
        plot_fif_comparison: Generate plots comparing preprocessed vs
            reconstructed .fif files (default: False).

    Example:
        >>> from zuna.pipeline import run_zuna_pipeline
        >>> run_zuna_pipeline(
        ...     input_dir="/data/eeg/raw_fif",
        ...     working_dir="/data/eeg/working",
        ...     target_channel_count=['AF3', 'AF4', 'F1', 'F2'],
        ...     bad_channels=['Cz'],
        ...     gpu_device=0,
        ...     plot_fif_comparison=True,
        ... )
    """

    # Validate inputs
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Setup working directory paths
    working_path = Path(working_dir)
    preprocessed_fif_dir = working_path / "1_fif_input"
    pt_input_dir = working_path / "2_pt_input"
    pt_output_dir = working_path / "3_pt_output"
    fif_output_dir = working_path / "4_fif_output"

    for d in [preprocessed_fif_dir, pt_input_dir, pt_output_dir, fif_output_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Preprocessing (.fif → .pt)
    print("[1/4] Preprocessing...", flush=True)
    zuna_preprocessing(
        input_dir=str(input_path),
        output_dir=str(pt_input_dir),
        save_preprocessed_fif=True,
        preprocessed_fif_dir=str(preprocessed_fif_dir),
        target_channel_count=target_channel_count,
        bad_channels=bad_channels,
    )

    # Step 2: Model Inference (.pt → .pt)
    print("[2/4] Model inference...", flush=True)
    zuna_inference(
        input_dir=str(pt_input_dir),
        output_dir=str(pt_output_dir),
        gpu_device=gpu_device,
    )

    # Step 3: Reconstruction (.pt → .fif)
    print("[3/4] Reconstructing FIF files...", flush=True)
    zuna_pt_to_fif(
        input_dir=str(pt_output_dir),
        output_dir=str(fif_output_dir),
    )

    # Step 4: Visualization (optional)
    if plot_pt_comparison or plot_fif_comparison:
        print("[4/4] Visualizing pipeline outputs...", flush=True)
        zuna_plot(
            input_dir=str(input_path),
            working_dir=str(working_path),
            plot_pt=plot_pt_comparison,
            plot_fif=plot_fif_comparison,
        )

    # Cleanup intermediate files if requested
    if not keep_intermediate_files:
        shutil.rmtree(pt_input_dir)
        shutil.rmtree(pt_output_dir)
        print("Removed intermediate PT files.")

    print("Pipeline complete. Output:", fif_output_dir)
