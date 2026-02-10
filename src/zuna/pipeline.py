"""
Zuna Complete Pipeline

This module provides a single function to run the complete EEG reconstruction pipeline:
.fif → .pt → model inference → .pt → .fif
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional

def run_zuna(
    input_dir: str,
    output_dir: str,
    checkpoint_path: str,
    upsample_factor: Optional[int] = None,
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
        tmp_dir: Temporary directory for intermediate .pt files
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

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Create directories - put tmp inside output_dir
    output_path = Path(output_dir)
    if tmp_dir is None:
        tmp_path = output_path / "tmp"
    else:
        tmp_path = Path(tmp_dir)

    pt_input_dir = tmp_path / "pt_input"
    pt_output_dir = tmp_path / "pt_output"

    for dir_path in [pt_input_dir, pt_output_dir, output_path]:
        dir_path.mkdir(parents=True, exist_ok=True)

    try:
        # =============================================================================
        # Step 1: Preprocess .fif → .pt
        # =============================================================================
        print("\n" + "="*80)
        print("STEP 1: Preprocessing .fif → .pt")
        print("="*80)

        from zuna import process_directory

        process_directory(
            input_dir=str(input_path),
            output_dir=str(pt_input_dir),
            target_sfreq=256.0,
            epoch_duration=5.0,
            apply_notch_filter=False  # Disable for short files
        )

        print(f"✓ Preprocessing complete")

        # =============================================================================
        # Step 2: Run Model Inference
        # =============================================================================
        print("\n" + "="*80)
        print("STEP 2: Running model inference")
        print("="*80)

        from omegaconf import OmegaConf
        import subprocess

        # Load the base config file
        config_path = Path(__file__).parent / "inference/AY2l/lingua/apps/AY2latent_bci/configs/config_bci_eval.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        # Load and modify config with our paths
        config = OmegaConf.load(str(config_path))
        config.data.data_dir = str(pt_input_dir.absolute())
        config.data.export_dir = str(pt_output_dir.absolute())
        config.checkpoint.init_ckpt_path = str(checkpoint.absolute())
        config.dump_dir = str(pt_output_dir.absolute())

        # Save modified config to temporary file
        temp_config_path = pt_output_dir / "temp_config.yaml"
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

        # Clean up temp config
        temp_config_path.unlink()

        print(f"✓ Inference complete")

        # =============================================================================
        # Step 3: Convert .pt → .fif (with optional upsampling)
        # =============================================================================
        print("\n" + "="*80)
        print("STEP 3: Converting .pt → .fif")
        if upsample_factor:
            print(f"  Upsampling factor: {upsample_factor}x")
        print("="*80)

        from zuna import pt_directory_to_fif

        # TODO: Add support for upsample_factor and data_key selection
        pt_directory_to_fif(
            input_dir=str(pt_output_dir),
            output_dir=str(output_path)
        )

        print(f"✓ Conversion complete")

        # =============================================================================
        # Cleanup
        # =============================================================================
        if cleanup_tmp:
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
        if cleanup_tmp and tmp_path.exists():
            print(f"\nCleaning up tmp directory after error...")
            shutil.rmtree(tmp_path)

        raise
