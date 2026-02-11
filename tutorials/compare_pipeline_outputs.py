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

NUM_SAMPLES = 2  # Number of random files to compare (set to 1 for quick check)
NORMALIZE_FOR_COMPARISON = False  # Normalize both to same scale for visual comparison

# Directory paths
# FIF_INPUT_DIR = "data/1_fif_input"
# FIF_OUTPUT_DIR = "data/4_fif_output"
# PT_INPUT_DIR = "data/2_pt_input"
# PT_OUTPUT_DIR = "data/3_pt_output"
# OUTPUT_DIR = "eval_figures"

FIF_INPUT_DIR = "/data/datasets/bci/dataset_downloads_cw/pip_test/1_fif_input_processed"   # Preprocessed .fif files (ground truth)
PT_INPUT_DIR = '/data/datasets/bci/dataset_downloads_cw/pip_test/2_pt_input'                  # Preprocessed .pt files
PT_OUTPUT_DIR = '/data/datasets/bci/dataset_downloads_cw/pip_test/3_pt_output'                # Model output .pt files
FIF_OUTPUT_DIR = "/data/datasets/bci/dataset_downloads_cw/pip_test/4_fif_output"              # Reconstructed .fif files
OUTPUT_DIR = "/data/datasets/bci/dataset_downloads_cw/pip_test/FIGURES"

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

    n_epochs = len(di['data'])
    print(f"\nTotal epochs: {n_epochs}")

    # Analyze ALL epochs to see which are valid, None, or zero
    valid_epochs = []
    none_epochs = []
    zero_epochs = []

    for i in range(n_epochs):
        recon_output = do['data'][i]

        if recon_output is None:
            none_epochs.append(i)
        elif isinstance(recon_output, (torch.Tensor, np.ndarray)):
            output_array = recon_output.numpy() if isinstance(recon_output, torch.Tensor) else recon_output
            if np.all(output_array == 0):
                zero_epochs.append(i)
            else:
                valid_epochs.append(i)

    print(f"  Valid epochs:  {len(valid_epochs)} ({100*len(valid_epochs)/n_epochs:.1f}%)")
    print(f"  None epochs:   {len(none_epochs)} ({100*len(none_epochs)/n_epochs:.1f}%) - filtered by model")
    print(f"  Zero epochs:   {len(zero_epochs)} ({100*len(zero_epochs)/n_epochs:.1f}%) - all zeros")

    if len(valid_epochs) == 0:
        print("\n⚠️  No valid epochs to plot (all are None or zero)!")
        return

    # Compute statistics for ALL valid epochs
    print(f"\nComputing metrics for all {len(valid_epochs)} valid epochs...")
    all_stds_input = []
    all_stds_output = []
    all_correlations = []
    all_mse = []

    for epoch_i in valid_epochs:
        orig = di['data'][epoch_i]
        recon = do['data'][epoch_i]

        # Convert to numpy
        if isinstance(orig, torch.Tensor):
            orig = orig.numpy()
        if isinstance(recon, torch.Tensor):
            recon = recon.numpy()

        # Compute std
        all_stds_input.append(orig.std())
        all_stds_output.append(recon.std())

        # Handle channel mismatch for metrics
        n_in = orig.shape[0]
        n_out = recon.shape[0]
        if n_out < n_in:
            padding = np.zeros((n_in - n_out, recon.shape[1]))
            recon = np.vstack([recon, padding])
        elif n_out > n_in:
            recon = recon[:n_in, :]

        # Compute metrics
        all_mse.append(np.mean((orig - recon) ** 2))

        # Compute correlation per channel
        epoch_corrs = []
        for ch in range(min(n_in, n_out)):
            corr = np.corrcoef(orig[ch], recon[ch])[0, 1]
            if not np.isnan(corr):
                epoch_corrs.append(corr)
        if epoch_corrs:
            all_correlations.append(np.mean(epoch_corrs))

    # Print summary statistics
    print(f"\nSummary across all {len(valid_epochs)} valid epochs:")
    print(f"  Input std:   mean={np.mean(all_stds_input):.6e}, range=[{np.min(all_stds_input):.6e}, {np.max(all_stds_input):.6e}]")
    print(f"  Output std:  mean={np.mean(all_stds_output):.6e}, range=[{np.min(all_stds_output):.6e}, {np.max(all_stds_output):.6e}]")
    print(f"  Correlation: mean={np.mean(all_correlations):.4f}, range=[{np.min(all_correlations):.4f}, {np.max(all_correlations):.4f}]")
    print(f"  MSE:         mean={np.mean(all_mse):.6e}, range=[{np.min(all_mse):.6e}, {np.max(all_mse):.6e}]")

    # Check if any turned to zero
    zero_std_count = sum(1 for s in all_stds_output if s < 1e-10)
    if zero_std_count > 0:
        print(f"  ⚠️  {zero_std_count} epochs have near-zero output std!")

    # Plot only 1 random valid epoch
    epoch_to_plot = random.choice(valid_epochs)
    print(f"\nPlotting 1 random valid epoch: {epoch_to_plot}")

    # Plot the selected epoch
    epoch_i = epoch_to_plot
    orig_input = di['data'][epoch_i]
    recon_output = do['data'][epoch_i]

    # Convert to numpy
    if isinstance(orig_input, torch.Tensor):
        orig_input = orig_input.numpy()
    if isinstance(recon_output, torch.Tensor):
        recon_output = recon_output.numpy()

    print(f"\nPlotting epoch {epoch_i}:")
    print(f"  Input shape:  {orig_input.shape}")
    print(f"  Output shape: {recon_output.shape}")

    # Handle channel count mismatch
    n_input_channels = orig_input.shape[0]
    n_output_channels = recon_output.shape[0]

    if n_output_channels != n_input_channels:
        print(f"  Channel mismatch: padding output from {n_output_channels} to {n_input_channels} channels")
        if n_output_channels < n_input_channels:
            # Pad with zeros
            padding = np.zeros((n_input_channels - n_output_channels, recon_output.shape[1]))
            recon_output = np.vstack([recon_output, padding])
        else:
            # Truncate
            recon_output = recon_output[:n_input_channels, :]

    # Compute metrics for this epoch
    mse = np.mean((orig_input - recon_output) ** 2)
    mae = np.mean(np.abs(orig_input - recon_output))

    correlations = []
    for ch in range(orig_input.shape[0]):
        if ch < n_output_channels:
            corr = np.corrcoef(orig_input[ch], recon_output[ch])[0, 1]
            correlations.append(corr)
        else:
            correlations.append(0.0)

    # Mean correlation only over non-padded channels
    non_zero_corrs = [c for ch_i, c in enumerate(correlations) if ch_i < n_output_channels]
    mean_corr = np.mean(non_zero_corrs) if len(non_zero_corrs) > 0 else 0.0

    print(f"\nEpoch {epoch_i} individual metrics:")
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

        # Mark zero-padded channels differently
        if ch < n_output_channels:
            ax.plot(time, recon_output[ch], 'r-', alpha=0.7, linewidth=0.8, label='Reconstructed')
        else:
            ax.plot(time, recon_output[ch], 'gray', alpha=0.3, linewidth=0.8, label='Zero-padded', linestyle='--')

        corr = correlations[ch]

        ch_label = f'Ch {ch}'
        if ch >= n_output_channels:
            ch_label = f'Ch {ch} (dropped)'

        ax.set_ylabel(ch_label, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

        if ch < n_output_channels:
            ax.text(0.98, 0.95, f'r={corr:.3f}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=7,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        else:
            ax.text(0.98, 0.95, 'r=N/A', transform=ax.transAxes,
                    ha='right', va='top', fontsize=7,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

        if ch == 0:
            ax.legend(fontsize=8, loc='upper left')
        if ch == num_channels - 1:
            ax.set_xlabel('Time (s)', fontsize=10)

    n_padded = n_input_channels - n_output_channels if n_output_channels < n_input_channels else 0
    title = f'PT File {file_idx} - Epoch {epoch_i}: Original vs Reconstructed\nMean Corr: {mean_corr:.4f}, MSE: {mse:.6f}'
    if n_padded > 0:
        title += f' ({n_padded} channels zero-padded)'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f"pt_file{file_idx}_epoch{epoch_i}_comparison.png"
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

    print(f"\nInput (before filtering):  {raw_input.info['nchan']} channels, {raw_input.n_times} samples, {raw_input.info['sfreq']} Hz")
    print(f"Output (before filtering): {raw_output.info['nchan']} channels, {raw_output.n_times} samples, {raw_output.info['sfreq']} Hz")

    # Filter to EEG channels only (exclude EOG, ECG, etc.)
    try:
        raw_input.pick_types(eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
        raw_output.pick_types(eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
        print(f"\nAfter EEG filtering:")
        print(f"  Input:  {raw_input.info['nchan']} EEG channels")
        print(f"  Output: {raw_output.info['nchan']} EEG channels")
    except Exception as e:
        print(f"  Warning: Could not filter to EEG channels: {e}")

    # Get data
    data_input = raw_input.get_data()  # (n_channels, n_times)
    data_output = raw_output.get_data()

    # Print data statistics BEFORE any processing
    print(f"\nData statistics (RAW - before padding):")
    print(f"  Input:  mean={data_input.mean():.6e}, std={data_input.std():.6e}, range=[{data_input.min():.6e}, {data_input.max():.6e}]")
    print(f"  Output: mean={data_output.mean():.6e}, std={data_output.std():.6e}, range=[{data_output.min():.6e}, {data_output.max():.6e}]")
    print(f"  Input in µV:  mean={data_input.mean()*1e6:.2f}, std={data_input.std()*1e6:.2f}")
    print(f"  Output in µV: mean={data_output.mean()*1e6:.2f}, std={data_output.std()*1e6:.2f}")

    # Handle channel count mismatch
    n_input_channels = data_input.shape[0]
    n_output_channels = data_output.shape[0]

    if n_output_channels < n_input_channels:
        print(f"  Padding output from {n_output_channels} to {n_input_channels} channels with zeros")
        # Pad output with zeros to match input channel count
        padding = np.zeros((n_input_channels - n_output_channels, data_output.shape[1]))
        data_output = np.vstack([data_output, padding])
    elif n_output_channels > n_input_channels:
        print(f"  Output has more channels ({n_output_channels}) than input ({n_input_channels})")
        print(f"  → Padding input with zeros to match output (channels {n_input_channels}-{n_output_channels-1} are model-generated)")
        # Pad INPUT with zeros to match output channel count
        padding = np.zeros((n_output_channels - n_input_channels, data_input.shape[1]))
        data_input = np.vstack([data_input, padding])

    # Take a 30-second window for visualization (longer for continuous data)
    sfreq = raw_input.info['sfreq']
    window_duration = 30.0  # seconds (increased from 5s)
    window_samples = int(window_duration * sfreq)

    # Pick a RANDOM 30-second window (instead of always the middle)
    min_samples = min(data_input.shape[1], data_output.shape[1])
    max_start = max(0, min_samples - window_samples)
    if max_start > 0:
        start_sample = random.randint(0, max_start)
        print(f"\nRandom window: {start_sample / sfreq:.1f}s - {(start_sample + window_samples) / sfreq:.1f}s")
    else:
        start_sample = 0
    end_sample = min(start_sample + window_samples, min_samples)
    actual_window_samples = end_sample - start_sample

    data_input_window = data_input[:, start_sample:end_sample]
    data_output_window = data_output[:, start_sample:end_sample]

    # Normalize for visual comparison if enabled
    if NORMALIZE_FOR_COMPARISON:
        print(f"\n⚠️  NORMALIZE_FOR_COMPARISON=True: Normalizing both to same scale (ignoring zeros)")

        # For output, compute stats ONLY on non-zero samples (ignore None epochs)
        output_nonzero_mask = data_output_window != 0
        output_nonzero_data = data_output_window[output_nonzero_mask]

        # Input stats (all data)
        input_mean = data_input_window[:n_output_channels].mean()
        input_std = data_input_window[:n_output_channels].std()

        # Output stats (only non-zero samples)
        if len(output_nonzero_data) > 0:
            output_mean = output_nonzero_data.mean()
            output_std = output_nonzero_data.std()
        else:
            output_mean = 0.0
            output_std = 1.0

        print(f"  Before normalization:")
        print(f"    Input:  mean={input_mean:.6e}, std={input_std:.6e}")
        print(f"    Output: mean={output_mean:.6e}, std={output_std:.6e} (non-zero only)")
        print(f"    Output: {len(output_nonzero_data)} / {output_nonzero_mask.size} non-zero samples ({100*len(output_nonzero_data)/output_nonzero_mask.size:.1f}%)")

        # Normalize input
        if input_std > 0:
            data_input_window_normalized = (data_input_window - input_mean) / input_std
        else:
            data_input_window_normalized = data_input_window.copy()

        # Normalize output (only non-zero samples)
        data_output_window_normalized = data_output_window.copy()
        if output_std > 0 and len(output_nonzero_data) > 0:
            data_output_window_normalized[output_nonzero_mask] = (output_nonzero_data - output_mean) / output_std

        # Use normalized data for plotting
        data_input_window_plot = data_input_window_normalized
        data_output_window_plot = data_output_window_normalized
        scale_label = "(normalized)"
    else:
        # Use original data converted to µV
        data_input_window_plot = data_input_window * 1e6
        data_output_window_plot = data_output_window * 1e6
        scale_label = "(µV)"

    # Compute metrics (only on non-zero channels in output)
    # Identify which channels have actual data (not just padding)
    non_zero_channels = []
    for ch in range(data_input_window.shape[0]):
        if ch < n_output_channels:
            non_zero_channels.append(ch)

    if len(non_zero_channels) > 0:
        mse = np.mean((data_input_window[non_zero_channels] - data_output_window[non_zero_channels]) ** 2)
        mae = np.mean(np.abs(data_input_window[non_zero_channels] - data_output_window[non_zero_channels]))
    else:
        mse = np.nan
        mae = np.nan

    correlations = []
    for ch in range(data_input_window.shape[0]):
        if ch < n_output_channels and np.any(data_output_window[ch] != 0):
            # Compute correlation for channels with actual data
            corr = np.corrcoef(data_input_window[ch], data_output_window[ch])[0, 1]
            correlations.append(corr)
        else:
            # Zero-padded channel - no correlation
            correlations.append(0.0)

    # Mean correlation only over non-zero channels
    non_zero_corrs = [c for i, c in enumerate(correlations) if i < n_output_channels]
    mean_corr = np.mean(non_zero_corrs) if len(non_zero_corrs) > 0 else 0.0

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
        time = np.arange(actual_window_samples) / sfreq

        # Mark model-generated channels differently (where input was padded with zeros)
        if ch < n_input_channels:
            # Real channel in both input and output
            ax.plot(time, data_input_window_plot[ch], 'b-', alpha=0.7, linewidth=0.8, label='Original')
            ax.plot(time, data_output_window_plot[ch], 'r-', alpha=0.7, linewidth=0.8, label='Reconstructed')
        else:
            # Model-generated channel (input was zero-padded)
            ax.plot(time, data_input_window_plot[ch], 'gray', alpha=0.3, linewidth=0.8, label='Input (zero-padded)', linestyle='--')
            ax.plot(time, data_output_window_plot[ch], 'r-', alpha=0.7, linewidth=0.8, label='Model-generated')

        corr = correlations[ch]

        ch_name = raw_input.ch_names[ch] if ch < len(raw_input.ch_names) else f'Ch {ch}'

        # Add indicator for model-generated channels
        if ch >= n_input_channels:
            ch_name = f'{ch_name} (model-generated)'

        ax.set_ylabel(f'{ch_name}\n{scale_label}', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

        # Show correlation text (or "N/A" for zero-padded)
        if ch < n_output_channels:
            ax.text(0.98, 0.95, f'r={corr:.3f}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=7,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        else:
            ax.text(0.98, 0.95, 'r=N/A', transform=ax.transAxes,
                    ha='right', va='top', fontsize=7,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

        if ch == 0:
            ax.legend(fontsize=8, loc='upper left')
        if ch == num_channels - 1:
            ax.set_xlabel('Time (s)', fontsize=10)

    n_padded = n_input_channels - n_output_channels if n_output_channels < n_input_channels else 0
    title = f'FIF File {file_idx}: Original vs Reconstructed ({window_duration:.0f}s window)\n'
    title += f'Mean Corr: {mean_corr:.4f}, MSE: {mse:.6e}'
    if n_padded > 0:
        title += f' ({n_padded} channels zero-padded)'
    plt.suptitle(title, fontsize=14, fontweight='bold')
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

    # Show PT file summary for ALL files
    if len(pt_input_files) > 0 and len(pt_output_files) > 0:
        print("\n" + "="*80)
        print("PT FILES EPOCH SUMMARY (All Files)")
        print("="*80)

        for pt_input in pt_input_files:
            # Find matching output
            pt_output = None
            for f in pt_output_files:
                if f.name == pt_input.name:
                    pt_output = f
                    break

            if pt_output is None:
                print(f"\n{pt_input.name}: NO OUTPUT FILE")
                continue

            # Load and analyze
            di = torch.load(pt_input, weights_only=False)
            do = torch.load(pt_output, weights_only=False)

            n_epochs = len(di['data'])
            valid_count = 0
            none_count = 0
            zero_count = 0

            for i in range(n_epochs):
                recon = do['data'][i]
                if recon is None:
                    none_count += 1
                elif isinstance(recon, (torch.Tensor, np.ndarray)):
                    arr = recon.numpy() if isinstance(recon, torch.Tensor) else recon
                    if np.all(arr == 0):
                        zero_count += 1
                    else:
                        valid_count += 1

            orig_filename = di.get('metadata', {}).get('original_filename', 'UNKNOWN')
            print(f"\n{pt_input.name}")
            print(f"  Original FIF: {orig_filename}")
            print(f"  Total: {n_epochs} | Valid: {valid_count} ({100*valid_count/n_epochs:.0f}%) | None: {none_count} ({100*none_count/n_epochs:.0f}%) | Zero: {zero_count}")

            if none_count > 0:
                print(f"  ⚠️  {none_count} epochs filtered by model (channel count mismatch)")

    # Compare .pt files FIRST (so we get these even if FIF crashes)
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

    # Compare .fif files (after PT files, in case this crashes)
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

    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print(f"Plots saved to: {output_dir}")
