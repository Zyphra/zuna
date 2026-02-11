"""
Zuna Complete Pipeline

This module provides functions to run the complete EEG reconstruction pipeline:
.fif → .pt → model inference → .pt → .fif

Each step can be run independently or as a complete pipeline.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional


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
    target_channel_count: Optional[int] = None,
) -> None:
    """
    Preprocess .fif files to .pt format.

    Args:
        input_dir: Directory containing input .fif files
        output_dir: Directory to save preprocessed .pt files
        target_sfreq: Target sampling frequency (default: 256.0 Hz)
        epoch_duration: Duration of each epoch in seconds (default: 5.0)
        apply_notch_filter: Whether to apply notch filter (default: False)
        apply_highpass_filter: Whether to apply highpass filter at 0.5 Hz (default: True)
        apply_average_reference: Whether to apply average reference (default: True)
        save_preprocessed_fif: Save preprocessed FIF for comparison (default: True)
        preprocessed_fif_dir: Where to save preprocessed FIF (default: input_dir/preprocessed)
        drop_bad_channels: Whether to detect and drop bad channels (default: False)
        drop_bad_epochs: Whether to drop bad epochs (default: False)
        zero_out_artifacts: Whether to zero out artifact samples (default: False)
        target_channel_count: Upsample to this many channels (None for no upsampling, e.g., 40, 64, 128)
    """
    from zuna import process_directory
    from pathlib import Path

    print("="*80)
    print("STEP 1: Preprocessing .fif → .pt")
    print("="*80)

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
        target_channel_count=target_channel_count
    )

    print(f"✓ Preprocessing complete")
    if save_preprocessed_fif and preprocessed_fif_dir:
        print(f"  Preprocessed FIF files saved to: {preprocessed_fif_dir}")


def zuna_inference(
    input_dir: str,
    output_dir: str,
    checkpoint_path: str,
    gpu_device: int = 0
) -> None:
    """
    Run model inference on .pt files.

    Args:
        input_dir: Directory containing preprocessed .pt files
        output_dir: Directory to save model output .pt files
        checkpoint_path: Path to model checkpoint
        gpu_device: GPU device ID (default: 0)
    """
    from omegaconf import OmegaConf
    import subprocess

    print("="*80)
    print("STEP 2: Running model inference")
    print("="*80)

    # Validate checkpoint
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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
    config.checkpoint.init_ckpt_path = str(checkpoint.absolute())
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

    print(f"Running: CUDA_VISIBLE_DEVICES={gpu_device} python3 {eeg_eval_script.name} config=...")

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
    upsample_factor: Optional[int] = None
) -> None:
    """
    Convert .pt files back to .fif format.

    Args:
        input_dir: Directory containing .pt files from model inference
        output_dir: Directory to save reconstructed .fif files
        upsample_factor: Upsampling factor (None for no upsampling)
    """
    from zuna import pt_directory_to_fif

    print("="*80)
    print("STEP 3: Converting .pt → .fif")
    if upsample_factor:
        print(f"  Upsampling factor: {upsample_factor}x")
    print("="*80)

    # TODO: Add support for upsample_factor and data_key selection
    results = pt_directory_to_fif(
        input_dir=input_dir,
        output_dir=output_dir
    )

    # Print results
    print(f"\n  Successful: {results['successful']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Total: {results['total']}")

    if results['errors']:
        print(f"\n  Errors:")
        for error in results['errors']:
            print(f"    {error['original_filename']}: {error['error']}")

    if results['successful'] > 0:
        print(f"\n✓ Conversion complete")
    else:
        print(f"\n⚠️  No files converted successfully")


def run_zuna(
    input_dir: str,
    working_dir: str,
    checkpoint_path: str,
    target_channel_count: Optional[int] = None,
    keep_intermediate_files: bool = True,
    gpu_device: int = 0,
    plot_pt_comparison: bool = False,
    plot_fif_comparison: bool = False,
):
    """
    Run the complete Zuna pipeline: .fif → .pt → model → .pt → .fif

    Args:
        input_dir: Directory containing input .fif files
        working_dir: Working directory where subdirectories will be created:
                    - 1_fif_input/preprocessed/ (preprocessed FIF files)
                    - 2_pt_input/ (preprocessed PT files)
                    - 3_pt_output/ (model output PT files)
                    - 4_fif_output/ (reconstructed FIF files)
        checkpoint_path: Path to model checkpoint
        target_channel_count: Target number of channels for upsampling (None for no upsampling, e.g., 40, 64, 128).
                             New channels are added with zeros for the model to interpolate.
        keep_intermediate_files: If False, deletes .pt files after reconstruction (default: True)
        gpu_device: GPU device ID (default: 0)
        plot_pt_comparison: Whether to plot .pt file comparisons (default: False)
        plot_fif_comparison: Whether to plot .fif file comparisons (default: False)

    Returns:
        None
    """

    print("="*80)
    print("ZUNA PIPELINE")
    print("="*80)
    print(f"Input:      {input_dir}")
    print(f"Working:    {working_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    if target_channel_count:
        print(f"Channels:   {target_channel_count} (upsampling)")
    print("="*80)

    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)

    # Validate inputs
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Setup working directory structure
    working_path = Path(working_dir)
    preprocessed_fif_dir = working_path / "1_fif_input" / "preprocessed"
    pt_input_path = working_path / "2_pt_input"
    pt_output_path = working_path / "3_pt_output"
    fif_output_path = working_path / "4_fif_output"

    # Create directories
    for dir_path in [preprocessed_fif_dir, pt_input_path, pt_output_path, fif_output_path]:
        dir_path.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Preprocessing
        print("\n[1/3] Preprocessing...")
        from zuna import process_directory
        process_directory(
            input_dir=str(input_path),
            output_dir=str(pt_input_path),
            save_preprocessed_fif=True,
            preprocessed_fif_dir=str(preprocessed_fif_dir),
            target_channel_count=target_channel_count,
        )

        # Step 2: Model Inference
        print("\n[2/3] Model inference...")
        zuna_inference(
            input_dir=str(pt_input_path),
            output_dir=str(pt_output_path),
            checkpoint_path=checkpoint_path,
            gpu_device=gpu_device
        )

        # Step 3: Reconstruction
        print("\n[3/3] Reconstructing FIF files...")
        zuna_pt_to_fif(
            input_dir=str(pt_output_path),
            output_dir=str(fif_output_path),
        )

        # Cleanup intermediate files if requested
        if not keep_intermediate_files:
            print("\nCleaning up intermediate files...")
            import shutil
            shutil.rmtree(pt_input_path)
            shutil.rmtree(pt_output_path)
            print(f"✓ Removed PT files")

        # Visualization
        if plot_pt_comparison or plot_fif_comparison:
            print("\nGenerating comparison plots...")
            from zuna.visualization import compare_pipeline_outputs
            compare_pipeline_outputs(
                working_dir=str(working_path),
                plot_pt=plot_pt_comparison,
                plot_fif=plot_fif_comparison,
            )

        print("\n" + "="*80)
        print("✓ PIPELINE COMPLETE!")
        print("="*80)
        print(f"Output: {fif_output_path}")
        print("="*80)

    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise
