#!/usr/bin/env python3
"""
Tutorial: Zuna Preprocessing Pipeline

This tutorial shows how to preprocess EEG data using Zuna.
"""
import zuna
from zuna.preprocessing import process_directory


# =============================================================================
# OPTION 1: Test with Synthetic Data (for quick testing)
# =============================================================================
# Uncomment this section to test with synthetic data first

# import numpy as np
# import mne
#
# print("="*80)
# print("Creating synthetic EEG data for testing...")
# print("="*80)
#
# sfreq = 500.0  # Hz
# duration = 60  # seconds
# n_channels = 64
#
# # Create synthetic data
# info = mne.create_info(
#     ch_names=[f'EEG{i:03d}' for i in range(n_channels)],
#     sfreq=sfreq,
#     ch_types='eeg'
# )
#
# data = np.random.randn(n_channels, int(sfreq * duration)) * 1e-6  # Realistic scale
# raw = mne.io.RawArray(data, info)
#
# # Set montage (REQUIRED!)
# montage = mne.channels.make_standard_montage('standard_1005')
# raw.set_montage(montage, match_case=False)
#
# # Process with defaults
# print("\nProcessing synthetic data...")
# metadata = zuna.raw_to_pt(raw, 'output_synthetic.pt')
#
# print(f"\nProcessing complete!")
# print(f"  Original epochs: {metadata['n_epochs_original']}")
# print(f"  Saved epochs: {metadata['n_epochs_saved']}")
# print(f"  Bad channels: {len(metadata['bad_channels'])}")
#
# # Reconstruct
# if metadata['n_epochs_saved'] > 0:
#     raw_reconstructed = zuna.pt_to_raw('output_synthetic.pt')
#     print(f"\nReconstructed: {len(raw_reconstructed.ch_names)} channels, "
#           f"{raw_reconstructed.times[-1]:.1f}s duration")


# =============================================================================
# OPTION 2: Process Your Own Data (production usage)
# =============================================================================

# Specify your input and output directories
INPUT_DIR = "/path/to/your/raw/files"      # Folder with .fif/.edf/etc files
OUTPUT_DIR = "/path/to/save/pt/files"      # Where to save .pt files

# Montage settings
# - If your files already have montages: set MONTAGE_NAME = None
# - If your files don't have montages: set to "standard_1005" (or other MNE montage name)
MONTAGE_NAME = "standard_1005"  # Use None if files already have montages

# Processing configuration
PROCESS_CONFIG = {
    'drop_bad_channels': True,
    'drop_bad_epochs': True,
    'apply_notch_filter': True,
    'apply_highpass_filter': True,
    'save_incomplete_batches': True,       # Save even if < 64 epochs
    'min_epochs_to_save': 1,               # Minimum epochs required
    'target_sfreq': 256.0,
    'epoch_duration': 5.0,
    'epochs_per_file': 64,
}


if __name__ == "__main__":
    # Uncomment and edit these paths for your data:
    # INPUT_DIR = "/data/home/jonas/my_eeg_data"
    # OUTPUT_DIR = "/data/home/jonas/processed_pt_files"
    # MONTAGE_NAME = None  # If your .fif files already have montages

    # Validate paths
    if INPUT_DIR == "/path/to/your/raw/files":
        print("⚠️  Please edit INPUT_DIR and OUTPUT_DIR in the script first!")
        print("\nSet INPUT_DIR to your folder containing raw EEG files")
        print("Set OUTPUT_DIR to where you want to save processed PT files")
        print("\nMontage settings:")
        print("  - If your files already have montages: MONTAGE_NAME = None")
        print("  - If they don't: MONTAGE_NAME = 'standard_1005'")
        print("\nTip: You can also uncomment the synthetic data section above to test first.")
    else:
        # Run batch processing
        # This function:
        # 1. Finds all EEG files in INPUT_DIR (.fif, .edf, .bdf, etc.)
        # 2. Checks if each file has a montage (uses MONTAGE_NAME if not)
        # 3. Processes each file with your config
        # 4. Saves PT files to OUTPUT_DIR
        results = process_directory(
            INPUT_DIR,
            OUTPUT_DIR,
            montage_name=MONTAGE_NAME,
            **PROCESS_CONFIG
        )

        # Optional: Test reconstruction on first successful file
        successful_results = [r for r in results if r['success']]
        if successful_results:
            print("\n" + "="*80)
            print("Testing Reconstruction")
            print("="*80)
            first_output = successful_results[0]['output']
            print(f"\nReconstructing: {first_output}")

            raw_reconstructed = zuna.pt_to_raw(first_output)
            print(f"  Channels: {len(raw_reconstructed.ch_names)}")
            print(f"  Duration: {raw_reconstructed.times[-1]:.1f} seconds")
            print(f"  Sampling rate: {raw_reconstructed.info['sfreq']} Hz")
