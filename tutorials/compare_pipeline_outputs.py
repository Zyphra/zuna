#!/usr/bin/env python3
"""
Visual Inspection Script for Zuna Pipeline

Compares input vs output for both .pt and .fif files:
- Randomly selects files from input/output directories
- Generates comparison plots showing original vs reconstructed signals
- Saves figures to tutorials/eval_figures/
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import mne
from pathlib import Path
import random

# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_SAMPLES = 1  # Number of random files to compare (set to 1 for quick check)

# Directory paths
FIF_INPUT_DIR = "data/1_fif_input"
FIF_OUTPUT_DIR = "data/4_fif_output"
PT_INPUT_DIR = "data/2_pt_input"
PT_OUTPUT_DIR = "data/3_pt_output"
OUTPUT_DIR = "eval_figures"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compare_pt_files(input_file, output_file, output_dir, file_idx):
    """Compare a single pair of .pt files (input vs output)"""

    print(f"\n{'='*80}")
    print(f"PT FILE COMPARISON {file_idx}")
    print(f"{'='*80}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")

    # Load files
    di = torch.load(input_file, weights_only=False)
    do = torch.load(output_file, weights_only=False)

    print(f"\nInput has {len(di['data'])} samples")
    print(f"Output has {len(do['data'])} samples")

    # Compare first valid sample
    num_samples_to_plot = min(1, len(di['data']))

    for i in range(num_samples_to_plot):
        orig_input = di['data'][i]
        recon_output = do['data'][i]

        if recon_output is None:
            print(f"WARNING: Sample {i} reconstruction is None (skipping)")
            continue

        # Convert to numpy
        if isinstance(orig_input, torch.Tensor):
            orig_input = orig_input.numpy()
        if isinstance(recon_output, torch.Tensor):
            recon_output = recon_output.numpy()

        # Compute metrics
        mse = np.mean((orig_input - recon_output) ** 2)
        mae = np.mean(np.abs(orig_input - recon_output))

        correlations = []
        for ch in range(orig_input.shape[0]):
            corr = np.corrcoef(orig_input[ch], recon_output[ch])[0, 1]
            correlations.append(corr)
        mean_corr = np.mean(correlations)

        print(f"\nSample {i} metrics:")
        print(f"  Shape: {orig_input.shape}")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Mean correlation: {mean_corr:.4f}")

        # Create plot
        num_channels, num_timepoints = orig_input.shape
        fig, axes = plt.subplots(num_channels, 1, figsize=(20, 2 * num_channels))
        if num_channels == 1:
            axes = [axes]

        for ch in range(num_channels):
            ax = axes[ch]
            time = np.arange(num_timepoints) / 256  # 256 Hz sampling rate

            ax.plot(time, orig_input[ch], 'b-', alpha=0.7, linewidth=0.8, label='Original')
            ax.plot(time, recon_output[ch], 'r-', alpha=0.7, linewidth=0.8, label='Reconstructed')

            corr = np.corrcoef(orig_input[ch], recon_output[ch])[0, 1]

            ax.set_ylabel(f'Ch {ch}', fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
            ax.text(0.98, 0.95, f'r={corr:.3f}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=7,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

            if ch == 0:
                ax.legend(fontsize=8, loc='upper left')
            if ch == num_channels - 1:
                ax.set_xlabel('Time (s)', fontsize=10)

        plt.suptitle(f'PT File {file_idx} - Sample {i}: Original vs Reconstructed\nMean Corr: {mean_corr:.4f}, MSE: {mse:.6f}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = output_dir / f"pt_file{file_idx}_sample{i}_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_path}")


def compare_fif_files(input_file, output_file, output_dir, file_idx):
    """Compare a single pair of .fif files (input vs output)"""

    print(f"\n{'='*80}")
    print(f"FIF FILE COMPARISON {file_idx}")
    print(f"{'='*80}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")

    # Load files
    raw_input = mne.io.read_raw_fif(input_file, preload=True, verbose=False)
    raw_output = mne.io.read_raw_fif(output_file, preload=True, verbose=False)

    print(f"\nInput:  {raw_input.info['nchan']} channels, {raw_input.n_times} samples, {raw_input.info['sfreq']} Hz")
    print(f"Output: {raw_output.info['nchan']} channels, {raw_output.n_times} samples, {raw_output.info['sfreq']} Hz")

    # Get data
    data_input = raw_input.get_data()  # (n_channels, n_times)
    data_output = raw_output.get_data()

    # Take a 5-second window for visualization
    sfreq = raw_input.info['sfreq']
    window_duration = 5.0  # seconds
    window_samples = int(window_duration * sfreq)

    # Use middle portion of the data
    start_sample = max(0, (data_input.shape[1] - window_samples) // 2)
    end_sample = start_sample + window_samples

    data_input_window = data_input[:, start_sample:end_sample]
    data_output_window = data_output[:, start_sample:end_sample]

    # Compute metrics
    mse = np.mean((data_input_window - data_output_window) ** 2)
    mae = np.mean(np.abs(data_input_window - data_output_window))

    correlations = []
    for ch in range(data_input_window.shape[0]):
        corr = np.corrcoef(data_input_window[ch], data_output_window[ch])[0, 1]
        correlations.append(corr)
    mean_corr = np.mean(correlations)

    print(f"\nMetrics (5s window):")
    print(f"  MSE: {mse:.6e}")
    print(f"  MAE: {mae:.6e}")
    print(f"  Mean correlation: {mean_corr:.4f}")
    print(f"  Correlation range: [{min(correlations):.4f}, {max(correlations):.4f}]")

    # Create plot
    num_channels = data_input_window.shape[0]
    fig, axes = plt.subplots(num_channels, 1, figsize=(20, 2 * num_channels))
    if num_channels == 1:
        axes = [axes]

    for ch in range(num_channels):
        ax = axes[ch]
        time = np.arange(window_samples) / sfreq

        ax.plot(time, data_input_window[ch] * 1e6, 'b-', alpha=0.7, linewidth=0.8, label='Original')  # Convert to µV
        ax.plot(time, data_output_window[ch] * 1e6, 'r-', alpha=0.7, linewidth=0.8, label='Reconstructed')

        corr = correlations[ch]

        ch_name = raw_input.ch_names[ch] if ch < len(raw_input.ch_names) else f'Ch {ch}'
        ax.set_ylabel(f'{ch_name}\n(µV)', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        ax.text(0.98, 0.95, f'r={corr:.3f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=7,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        if ch == 0:
            ax.legend(fontsize=8, loc='upper left')
        if ch == num_channels - 1:
            ax.set_xlabel('Time (s)', fontsize=10)

    plt.suptitle(f'FIF File {file_idx}: Original vs Reconstructed (5s window)\nMean Corr: {mean_corr:.4f}, MSE: {mse:.6e}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f"fif_file{file_idx}_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("ZUNA PIPELINE VISUAL INSPECTION")
    print("="*80)
    print(f"Comparing {NUM_SAMPLES} random file(s) from input/output directories")
    print()

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of files
    fif_input_files = sorted(Path(FIF_INPUT_DIR).glob("*.fif"))
    fif_output_files = sorted(Path(FIF_OUTPUT_DIR).glob("*.fif"))
    pt_input_files = sorted(Path(PT_INPUT_DIR).glob("*.pt"))
    pt_output_files = sorted(Path(PT_OUTPUT_DIR).glob("*.pt"))

    print(f"Found {len(fif_input_files)} .fif input files")
    print(f"Found {len(fif_output_files)} .fif output files")
    print(f"Found {len(pt_input_files)} .pt input files")
    print(f"Found {len(pt_output_files)} .pt output files")

    # Compare .fif files
    if len(fif_input_files) > 0 and len(fif_output_files) > 0:
        # Randomly sample files
        sample_indices = random.sample(range(min(len(fif_input_files), len(fif_output_files))),
                                      min(NUM_SAMPLES, len(fif_input_files), len(fif_output_files)))

        for idx, file_idx in enumerate(sample_indices):
            input_file = fif_input_files[file_idx]
            # Match output file by name
            output_file = None
            for f in fif_output_files:
                if f.stem == input_file.stem or input_file.stem in f.stem:
                    output_file = f
                    break

            if output_file is None:
                print(f"\nWarning: No matching output file found for {input_file.name}")
                continue

            compare_fif_files(input_file, output_file, output_dir, idx + 1)
    else:
        print("\nSkipping .fif comparison (no files found)")

    # Compare .pt files
    if len(pt_input_files) > 0 and len(pt_output_files) > 0:
        # Randomly sample files
        sample_indices = random.sample(range(min(len(pt_input_files), len(pt_output_files))),
                                      min(NUM_SAMPLES, len(pt_input_files), len(pt_output_files)))

        for idx, file_idx in enumerate(sample_indices):
            input_file = pt_input_files[file_idx]
            # Match output file by name
            output_file = None
            for f in pt_output_files:
                if f.name == input_file.name:
                    output_file = f
                    break

            if output_file is None:
                print(f"\nWarning: No matching output file found for {input_file.name}")
                continue

            compare_pt_files(input_file, output_file, output_dir, idx + 1)
    else:
        print("\nSkipping .pt comparison (no files found)")

    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print(f"Plots saved to: {output_dir}")
