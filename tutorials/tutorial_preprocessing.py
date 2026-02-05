#!/usr/bin/env python3
"""
Tutorial: Zuna Preprocessing Pipeline

This tutorial shows how to preprocess EEG data using Zuna.
"""
import zuna
from zuna.preprocessing import process_directory

INPUT_DIR = "data/1_fif_input"      # Folder with raw .fif files
OUTPUT_DIR = "data/2_pt_input"      # Where to save .pt files

# IMPORTANT: Your input files MUST already have montages set with 3D channel positions
# See MNE for further infos: https://mne.tools/stable/generated/mne.channels.make_standard_montage.html
# Files without montages will be automatically skipped during processing

# Processing configuration
PROCESS_CONFIG = {
    'drop_bad_channels': False,
    'drop_bad_epochs': False,
    'apply_notch_filter': False,
    'apply_highpass_filter': True,
    'save_incomplete_batches': True,       # Save even if < 64 epochs
    'min_epochs_to_save': 1,               # Minimum epochs required
    'target_sfreq': 256.0,
    'epoch_duration': 5.0,
    'epochs_per_file': 64,
    'max_duration_minutes': 10.0,          # Split files longer than 10 minutes into chunks
}

# Parallel processing settings
N_JOBS = 10  # Set to > 1 for parallel processing (e.g., 4, 8) or -1 for all CPUs


if __name__ == "__main__":
    # Process directory
    # - n_jobs=1: Sequential processing (default)
    # - n_jobs=4: Process 4 files in parallel
    # - n_jobs=-1: Use all available CPU cores
    results = process_directory(
        INPUT_DIR,
        OUTPUT_DIR,
        n_jobs=N_JOBS,
        **PROCESS_CONFIG
    )
    # Optional: Test reconstruction on first successful file
    successful_results = [r for r in results if r['success']]

    if successful_results:
        print("\n" + "="*80)
        print("Testing Reconstruction")
        print("="*80)

        # Get first output file (handle chunked files)
        first_result = successful_results[0]
        if 'outputs' in first_result:
            first_output = first_result['outputs'][0]  # First chunk if file was split
        else:
            first_output = first_result['output']

        print(f"\nReconstructing: {first_output}")

        raw_reconstructed = zuna.pt_to_raw(first_output)
        print(f"  Channels: {len(raw_reconstructed.ch_names)}")
        print(f"  Duration: {raw_reconstructed.times[-1]:.1f} seconds")
        print(f"  Sampling rate: {raw_reconstructed.info['sfreq']} Hz")
