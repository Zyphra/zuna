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
    save_preprocessed_fif: bool = True,
    preprocessed_fif_dir: Optional[str] = None,
    drop_bad_channels: bool = False,
    drop_bad_epochs: bool = False,
    zero_out_artifacts: bool = False,
) -> None:
    """
    Preprocess .fif files to .pt format.

    Args:
        input_dir: Directory containing input .fif files
        output_dir: Directory to save preprocessed .pt files
        target_sfreq: Target sampling frequency (default: 256.0 Hz)
        epoch_duration: Duration of each epoch in seconds (default: 5.0)
        apply_notch_filter: Whether to apply notch filter (default: False)
        save_preprocessed_fif: Save preprocessed FIF for comparison (default: True)
        preprocessed_fif_dir: Where to save preprocessed FIF (default: input_dir/preprocessed)
        drop_bad_channels: Whether to detect and drop bad channels (default: False)
        drop_bad_epochs: Whether to drop bad epochs (default: False)
        zero_out_artifacts: Whether to zero out artifact samples (default: False)
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
        save_preprocessed_fif=save_preprocessed_fif,
        preprocessed_fif_dir=preprocessed_fif_dir,
        drop_bad_channels=drop_bad_channels,
        drop_bad_epochs=drop_bad_epochs,
        zero_out_artifacts=zero_out_artifacts
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
    output_dir: str,
    checkpoint_path: str,
    upsample_factor: Optional[int] = None,
    pt_input_dir: Optional[str] = None,
    pt_output_dir: Optional[str] = None,
    tmp_dir: Optional[str] = None,  # Auto-creates in output_dir/tmp if None
    gpu_device: int = 0,
    cleanup_tmp: bool = True
):
    """
    Run the complete Zuna pipeline: .fif → latents → reconstructed .fif

    Args:
        input_dir: Directory containing input .fif files
        output_dir: Directory to save final reconstructed .fif files
        checkpoint_path: Path to model checkpoint
        upsample_factor: Upsampling factor (None for no upsampling)
        pt_input_dir: Custom path for preprocessed .pt files (None = auto tmp)
        pt_output_dir: Custom path for model output .pt files (None = auto tmp)
        tmp_dir: Temporary directory (ignored if pt_input_dir/pt_output_dir specified)
        gpu_device: GPU device ID (default: 0)
        cleanup_tmp: Whether to delete tmp directory after completion (default: True)

    Returns:
        None
    """

    print("="*80)
    print("ZUNA COMPLETE PIPELINE")
    print("="*80)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Upsample: {upsample_factor if upsample_factor else 'None'}")
    print(f"GPU: {gpu_device}")
    print("="*80)

    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)

    # Validate inputs
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Setup paths for intermediate files
    output_path = Path(output_dir)

    # Use custom paths if provided, otherwise create tmp directories
    if pt_input_dir is not None or pt_output_dir is not None:
        # Custom paths specified
        pt_input_path = Path(pt_input_dir) if pt_input_dir else output_path / "tmp" / "pt_input"
        pt_output_path = Path(pt_output_dir) if pt_output_dir else output_path / "tmp" / "pt_output"
        tmp_path = None  # Don't cleanup custom directories
    else:
        # Auto tmp directories
        if tmp_dir is None:
            tmp_path = output_path / "tmp"
        else:
            tmp_path = Path(tmp_dir)
        pt_input_path = tmp_path / "pt_input"
        pt_output_path = tmp_path / "pt_output"

    # Create directories
    for dir_path in [pt_input_path, pt_output_path, output_path]:
        dir_path.mkdir(parents=True, exist_ok=True)

    try:
        # Setup preprocessed FIF directory (for ground truth comparison)
        # Create a sibling folder at the same level as input, with _processed suffix
        # E.g., if input is "1_fif_input", create "1_fif_input_processed"
        preprocessed_fif_dir = input_path.parent / f"{input_path.name}_processed"

        # Step 1: Preprocessing (with preprocessed FIF saving)
        print()
        from zuna import process_directory
        process_directory(
            input_dir=str(input_path),
            output_dir=str(pt_input_path),
            target_sfreq=256.0,
            epoch_duration=5.0,
            apply_notch_filter=False,  # Disable for short files
            save_preprocessed_fif=True,  # Save for comparison
            preprocessed_fif_dir=str(preprocessed_fif_dir),
            drop_bad_channels=False,  # Keep all channels (no removal)
            drop_bad_epochs=False,    # Keep all epochs (no removal)
            zero_out_artifacts=False, # Keep all data (no zeroing)
        )
        print(f"✓ Preprocessing complete")
        print(f"  Preprocessed FIF files saved to: {preprocessed_fif_dir}")

        # Step 2: Model Inference
        print()
        zuna_inference(
            input_dir=str(pt_input_path),
            output_dir=str(pt_output_path),
            checkpoint_path=checkpoint_path,
            gpu_device=gpu_device
        )

        # Step 3: PT to FIF conversion
        print()
        zuna_pt_to_fif(
            input_dir=str(pt_output_path),
            output_dir=str(output_path),
            upsample_factor=upsample_factor
        )

        # Cleanup
        if cleanup_tmp and tmp_path is not None:
            print("\n" + "="*80)
            print("Cleaning up temporary files...")
            print("="*80)
            shutil.rmtree(tmp_path)
            print(f"✓ Removed: {tmp_path}")

        print("\n" + "="*80)
        print("✓ PIPELINE COMPLETE!")
        print("="*80)
        print(f"Final output: {output_path}")
        print("="*80)

    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

        # Optionally cleanup on failure too
        if cleanup_tmp and tmp_path is not None and tmp_path.exists():
            print(f"\nCleaning up tmp directory after error...")
            shutil.rmtree(tmp_path)

        raise
