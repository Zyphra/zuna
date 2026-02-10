#!/usr/bin/env python3
"""
Tutorial: Zuna Inference Pipeline

This tutorial runs model inference on preprocessed PT files by calling
the eeg_eval.py script with a modified config file.

Usage:
    python 2_tutorial_inference.py

Requirements:
    - Preprocessed PT files in data/2_pt_input/
    - Model checkpoint at specified path
"""

import os
import sys
import subprocess
from pathlib import Path
from omegaconf import OmegaConf

# =============================================================================
# Configuration
# =============================================================================

INPUT_DIR = "data/2_pt_input"    # Folder with preprocessed .pt files
OUTPUT_DIR = "data/3_pt_output"  # Where to save model outputs
CHECKPOINT_PATH = "/data/checkpoints/bci/bci_AY2l_bigrun16e/checkpoints/0000150000"
GPU_DEVICE = 1  # Which GPU to use

# =============================================================================
# Run Inference
# =============================================================================

def main():
    print("="*80)
    print("Zuna EEG Inference Pipeline")
    print("="*80)

    # Validate paths
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        print(f"Error: Input directory '{INPUT_DIR}' not found!")
        print(f"Please run the preprocessing tutorial first (1_tutorial_preprocessing.py)")
        return

    checkpoint = Path(CHECKPOINT_PATH)
    if not checkpoint.exists():
        print(f"Error: Model checkpoint not found at '{CHECKPOINT_PATH}'!")
        return

    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nInput directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"GPU: {GPU_DEVICE}")

    # Load the base config file
    config_path = Path(__file__).parent.parent / "src/zuna/inference/AY2l/lingua/apps/AY2latent_bci/configs/config_bci_eval.yaml"

    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return

    # Load and modify config with our paths
    config = OmegaConf.load(str(config_path))
    config.data.data_dir = str(input_path.absolute())
    config.data.export_dir = str(output_path.absolute())
    config.checkpoint.init_ckpt_path = str(checkpoint.absolute())
    config.dump_dir = str(output_path.absolute())

    # Save modified config to temporary file
    temp_config_path = output_path / "temp_config.yaml"
    OmegaConf.save(config, str(temp_config_path))

    # Build command to run eeg_eval.py
    eeg_eval_script = Path(__file__).parent.parent / "src/zuna/inference/AY2l/lingua/apps/AY2latent_bci/eeg_eval.py"

    cmd = [
        "python3",
        str(eeg_eval_script),
        f"config={temp_config_path}"
    ]

    # Set environment variable for GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(GPU_DEVICE)

    print("\nRunning inference...")
    print(f"Command: CUDA_VISIBLE_DEVICES={GPU_DEVICE} python3 {eeg_eval_script} config={temp_config_path}")
    print("(This may take a few minutes depending on the number of files)\n")

    try:
        # Run the command
        result = subprocess.run(cmd, env=env, check=True)

        print("\n" + "="*80)
        print("Inference completed successfully!")
        print(f"Results saved to: {OUTPUT_DIR}")
        print("="*80)

        # Clean up temp config
        temp_config_path.unlink()

    except subprocess.CalledProcessError as e:
        print(f"\nError during inference: {e}")
        print("\nIf you encounter issues:")
        print("1. Make sure the checkpoint path is correct")
        print("2. Ensure PT files in input directory are properly preprocessed")
        print("3. Check that you have enough GPU memory")


if __name__ == "__main__":
    main()
