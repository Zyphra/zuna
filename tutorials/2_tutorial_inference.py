#!/usr/bin/env python3
"""
Tutorial: Zuna Inference Pipeline

This tutorial shows how to run the Zuna EEG foundation model to extract
latent representations and reconstruct EEG signals.

Usage:
    CUDA_VISIBLE_DEVICES=0 python 2_tutorial_inference.py

Requirements:
    - Preprocessed PT files in data/2_pt_input/
    - Model checkpoint (download or path to trained model)
"""




import json
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load

from lingua.args import dataclass_from_dict
from apps.AY2latent_bci.transformer import EncoderDecoder, DecoderTransformerArgs

# In your shell, set your HF_TOKEN environment variable: 
# export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"


REPO_ID = "Zyphra/ZUNA"
WEIGHTS = "model-00001-of-00001.safetensors"
CONFIG  = "config.json"  

# model arch
config_path = hf_hub_download(repo_id=REPO_ID, filename=CONFIG, token=True)
with open(config_path, "r") as f:
    config_dict = json.load(f)

# build model
model_args = dataclass_from_dict(DecoderTransformerArgs, config_dict["model"])
model = EncoderDecoder(model_args)

# download weights, load them into EncoderDecoder
weights_path = hf_hub_download(repo_id=REPO_ID, filename=WEIGHTS, token=True)
state_dict = safe_load(weights_path, device="cpu")

# remove .model prefix from keys
state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}

model.load_state_dict(state_dict, strict=True)
model.eval()



import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

################################################################################
################################################################################    

import sys
import os
from pathlib import Path

# Add the inference code to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src/zuna/inference/AY2l/lingua"))

from apps.AY2latent_bci import eeg_eval
from omegaconf import OmegaConf

# =============================================================================
# Configuration
# =============================================================================

# Input/Output directories
INPUT_DIR = "data/2_pt_input"    # Folder with preprocessed .pt files
OUTPUT_DIR = "data/3_pt_output"  # Where to save latent representations

# Model checkpoint path (update this to your checkpoint location)
CHECKPOINT_PATH = "/data/checkpoints/bci/bci_AY2l_bigrun16e/checkpoints/0000150000"

# Create a minimal config
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
        'use_b2': False,  # Use local filesystem, not cloud storage
        'data_dir': INPUT_DIR,
        'glob_filter': "**/*.pt",
        'sample_rate': 256,
        'seq_len': 1280,
        'num_fine_time_pts': 32,
        'use_coarse_time': "B",
        'cat_chan_xyz_and_eeg': False,
        'randomly_permute_sequence': False,
        'channel_dropout_prob': 0.50,
        'stft_global_sigma': 0.1,
        'num_bins_discretize_xyz_chan_pos': 50,
        'chan_pos_xyz_extremes_type': "twelves",
        'batch_size': 1,
        'target_packed_seqlen': 3000,
        'num_workers': 0,
        'prefetch_factor': None,
        'persistent_workers': False,
        'pin_memory': False,
        'diffusion_forcing': False,
        'shuffle': False,
        'seed': 316,
    },
    'checkpoint': {
        'init_ckpt_path': CHECKPOINT_PATH,
    },
    'dump_dir': OUTPUT_DIR,
    'name': 'bci_inference',
    'seed': 42,
    'steps': 1,
}

# =============================================================================
# Run Inference
# =============================================================================

def main():
    print("="*80)
    print("Zuna EEG Inference Pipeline")
    print("="*80)

    # Check if input directory exists
    if not Path(INPUT_DIR).exists():
        print(f"Error: Input directory '{INPUT_DIR}' not found!")
        print(f"Please run the preprocessing tutorial first (1_tutorial_preprocessing.py)")
        return

    # Check if checkpoint exists
    if not Path(CHECKPOINT_PATH).exists():
        print(f"Error: Model checkpoint not found at '{CHECKPOINT_PATH}'!")
        print(f"Please download or specify the correct path to the model checkpoint.")
        return

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"\nInput directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    # Convert config dict to OmegaConf and run evaluation
    config = OmegaConf.create(config_dict)
    config = OmegaConf.to_object(config)

    print("\nLoading model and running inference...")
    print("(This may take a few minutes depending on the number of files)")

    try:
        eeg_eval.evaluate(config)
        print("\n" + "="*80)
        print("Inference completed successfully!")
        print(f"Results saved to: {OUTPUT_DIR}")
        print("="*80)
    except Exception as e:
        print(f"\nError during inference: {e}")
        print("\nIf you encounter issues:")
        print("1. Make sure the checkpoint path is correct")
        print("2. Ensure PT files in input directory are properly preprocessed")
        print("3. Check that you have enough GPU memory")


if __name__ == "__main__":
    main()
