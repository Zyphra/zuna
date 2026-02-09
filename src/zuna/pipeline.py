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

        # Add inference code to path - need to add both the main path and the apps directory
        inference_path = Path(__file__).parent / "inference/AY2l/lingua"
        apps_path = inference_path / "apps/AY2latent_bci"
        sys.path.insert(0, str(inference_path))
        sys.path.insert(0, str(apps_path))

        from apps.AY2latent_bci import eeg_eval
        from omegaconf import OmegaConf

        config_dict = {
            'model': {
                'dim': 1024,
                'n_layers': 16,
                'head_dim': 64,
                'input_dim': 32,
                'encoder_input_dim': 32,
                'encoder_output_dim': 32,
                'encoder_latent_downsample_factor': 1,
                'sliding_window': 65536,
                'encoder_sliding_window': 65536,
                'xattn_sliding_window': 65536,
                'max_seqlen': 50,
                'num_fine_time_pts': 32,
                'rope_dim': 4,
                'rope_theta': 10000.0,
                'tok_idx_type': "{x,y,z,tc}",
                'stft_global_sigma': 0.1,
                'dropout_type': "zeros",
            },
            'data': {
                'use_b2': False,
                'data_dir': str(pt_input_dir),
                'export_dir': str(pt_output_dir),
                'glob_filter': "**/*.pt",
                'sample_rate': 256,
                'seq_len': 1280,
                'num_fine_time_pts': 32,
                'use_coarse_time': "B",
                'cat_chan_xyz_and_eeg': False,
                'randomly_permute_sequence': False,
                'channel_dropout_prob': 0.0,
                'stft_global_sigma': 0.1,
                'num_bins_discretize_xyz_chan_pos': 50,
                'chan_pos_xyz_extremes_type': "twelves",
                'batch_size': 1,
                'target_packed_seqlen': 1,
                'num_workers': 0,
                'prefetch_factor': None,
                'persistent_workers': False,
                'pin_memory': False,
                'diffusion_forcing': False,
                'shuffle': False,
                'seed': 316,
            },
            'checkpoint': {
                'init_ckpt_path': str(checkpoint),
            },
            'dump_dir': str(pt_output_dir),
            'name': 'zuna_pipeline',
        }

        # Keep as OmegaConf object (don't convert to dict)
        config = OmegaConf.create(config_dict)

        eeg_eval.evaluate(config)

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
