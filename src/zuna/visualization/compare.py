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
# HELPER FUNCTIONS
# =============================================================================

def compare_pt_files(input_file, output_file, output_dir, file_idx):
    """Compare a single pair of .pt files (input vs output)"""

    # Load files
    di = torch.load(input_file, weights_only=False)
    do = torch.load(output_file, weights_only=False)

    n_epochs = len(di['data'])

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

    if len(valid_epochs) == 0:
        return

    # Compute statistics for ALL valid epochs
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

    # Check if any turned to zero
    zero_std_count = sum(1 for s in all_stds_output if s < 1e-10)

    # Plot only 1 random valid epoch
    epoch_to_plot = random.choice(valid_epochs)

    # Plot the selected epoch
    epoch_i = epoch_to_plot
    orig_input = di['data'][epoch_i]
    recon_output = do['data'][epoch_i]

    # Convert to numpy
    if isinstance(orig_input, torch.Tensor):
        orig_input = orig_input.numpy()
    if isinstance(recon_output, torch.Tensor):
        recon_output = recon_output.numpy()

    # Handle channel count mismatch
    n_input_channels = orig_input.shape[0]
    n_output_channels = recon_output.shape[0]

    if n_output_channels != n_input_channels:
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

def compare_fif_files(original_file, preprocessed_file, output_file, output_dir, file_idx, include_original_fif, normalize_for_comparison):
    """Compare .fif files: preprocessed vs reconstructed (and optionally original if provided)"""

    # Check if original file is provided (3-line mode vs 2-line mode)
    include_original = include_original_fif and original_file is not None

    # Load files
    if include_original:
        raw_original = mne.io.read_raw_fif(original_file, preload=True, verbose=False)
    raw_preprocessed = mne.io.read_raw_fif(preprocessed_file, preload=True, verbose=False)
    raw_output = mne.io.read_raw_fif(output_file, preload=True, verbose=False)

    # Filter to EEG channels only (exclude EOG, ECG, etc.)
    try:
        if include_original:
            raw_original.pick_types(eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
        raw_preprocessed.pick_types(eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
        raw_output.pick_types(eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
    except Exception as e:
        pass  # Could not filter to EEG channels

    # Get data
    if include_original:
        data_original = raw_original.get_data()  # (n_channels, n_times)
    data_preprocessed = raw_preprocessed.get_data()
    data_output = raw_output.get_data()

    # Handle channel count mismatch
    n_input_channels = data_preprocessed.shape[0]
    n_output_channels = data_output.shape[0]

    if n_output_channels < n_input_channels:
        # Pad output with zeros to match input channel count
        padding = np.zeros((n_input_channels - n_output_channels, data_output.shape[1]))
        data_output = np.vstack([data_output, padding])
    elif n_output_channels > n_input_channels:
        # Pad INPUT with zeros to match output channel count
        padding = np.zeros((n_output_channels - n_input_channels, data_preprocessed.shape[1]))
        data_preprocessed = np.vstack([data_preprocessed, padding])

    # Take 30-second windows for visualization
    sfreq = raw_preprocessed.info['sfreq']
    window_duration = 30.0  # seconds
    window_samples = int(window_duration * sfreq)
    min_samples = min(data_preprocessed.shape[1], data_output.shape[1])

    # Generate TWO plots: beginning and end
    windows_to_plot = [
        ('beg', 0, min(window_samples, min_samples)),
        ('end', max(0, min_samples - window_samples), min_samples)
    ]

    for window_name, start_sample, end_sample in windows_to_plot:
        actual_window_samples = end_sample - start_sample

        if include_original:
            data_original_window = data_original[:, start_sample:end_sample]
        data_preprocessed_window = data_preprocessed[:, start_sample:end_sample]  # Preprocessed data
        data_output_window = data_output[:, start_sample:end_sample]

        # Normalize for visual comparison if enabled
        if normalize_for_comparison:
            # For output, compute stats ONLY on non-zero samples (ignore None epochs)
            output_nonzero_mask = data_output_window != 0
            output_nonzero_data = data_output_window[output_nonzero_mask]

            # Input stats (all data)
            input_mean = data_preprocessed_window[:n_output_channels].mean()
            input_std = data_preprocessed_window[:n_output_channels].std()

            # Output stats (only non-zero samples)
            if len(output_nonzero_data) > 0:
                output_mean = output_nonzero_data.mean()
                output_std = output_nonzero_data.std()
            else:
                output_mean = 0.0
                output_std = 1.0

            # Normalize input
            if input_std > 0:
                data_preprocessed_window_normalized = (data_preprocessed_window - input_mean) / input_std
            else:
                data_preprocessed_window_normalized = data_preprocessed_window.copy()

            # Normalize output (only non-zero samples)
            data_output_window_normalized = data_output_window.copy()
            if output_std > 0 and len(output_nonzero_data) > 0:
                data_output_window_normalized[output_nonzero_mask] = (output_nonzero_data - output_mean) / output_std

            # Use normalized data for plotting
            if include_original:
                data_original_window_plot = data_original_window * 1e6  # Original in µV (not normalized)
            data_preprocessed_window_plot = data_preprocessed_window_normalized
            data_output_window_plot = data_output_window_normalized
            scale_label = "(normalized)"
        else:
            # Use original data converted to µV
            if include_original:
                data_original_window_plot = data_original_window * 1e6
            data_preprocessed_window_plot = data_preprocessed_window * 1e6
            data_output_window_plot = data_output_window * 1e6
            scale_label = "(µV)"

        # Compute metrics (only on non-zero channels in output)
        # Identify which channels have actual data (not just padding)
        non_zero_channels = []
        for ch in range(data_preprocessed_window.shape[0]):
            if ch < n_output_channels:
                non_zero_channels.append(ch)

        if len(non_zero_channels) > 0:
            mse = np.mean((data_preprocessed_window[non_zero_channels] - data_output_window[non_zero_channels]) ** 2)
            mae = np.mean(np.abs(data_preprocessed_window[non_zero_channels] - data_output_window[non_zero_channels]))
        else:
            mse = np.nan
            mae = np.nan

        correlations = []
        for ch in range(data_preprocessed_window.shape[0]):
            if ch < n_output_channels and np.any(data_output_window[ch] != 0):
                # Compute correlation for channels with actual data
                corr = np.corrcoef(data_preprocessed_window[ch], data_output_window[ch])[0, 1]
                correlations.append(corr)
            else:
                # Zero-padded channel - no correlation
                correlations.append(0.0)

        # Mean correlation only over non-zero channels
        non_zero_corrs = [c for i, c in enumerate(correlations) if i < n_output_channels]
        mean_corr = np.mean(non_zero_corrs) if len(non_zero_corrs) > 0 else 0.0

        # Create plot
        num_channels = data_preprocessed_window.shape[0]
        fig, axes = plt.subplots(num_channels, 1, figsize=(20, 2 * num_channels))
        if num_channels == 1:
            axes = [axes]

        for ch in range(num_channels):
            ax = axes[ch]
            time = np.arange(actual_window_samples) / sfreq

            # Plot lines: Preprocessed vs Reconstructed (and optionally Original)
            if ch < n_input_channels:
                # Real channel
                if include_original:
                    ax.plot(time, data_original_window_plot[ch], 'gray', alpha=0.5, linewidth=0.6, label='Original', linestyle=':')
                ax.plot(time, data_preprocessed_window_plot[ch], 'b-', alpha=0.7, linewidth=0.8, label='Preprocessed')
                ax.plot(time, data_output_window_plot[ch], 'r-', alpha=0.7, linewidth=0.8, label='Reconstructed')
            else:
                # Model-generated channel (input was zero-padded)
                if include_original:
                    ax.plot(time, data_original_window_plot[ch], 'gray', alpha=0.3, linewidth=0.6, label='Original', linestyle=':')
                ax.plot(time, data_preprocessed_window_plot[ch], 'gray', alpha=0.3, linewidth=0.8, label='Preprocessed (zero-padded)', linestyle='--')
                ax.plot(time, data_output_window_plot[ch], 'r-', alpha=0.7, linewidth=0.8, label='Model-generated')

            corr = correlations[ch]

            ch_name = raw_preprocessed.ch_names[ch] if ch < len(raw_preprocessed.ch_names) else f'Ch {ch}'

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

        output_path = output_dir / f"fif_file{file_idx}_comparison_{window_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    # =========================================================================
    # Generate third plot: Full duration, single channel (first channel)
    # =========================================================================
    channel_idx = 0  # First channel
    ch_name = raw_preprocessed.ch_names[channel_idx] if channel_idx < len(raw_preprocessed.ch_names) else f'Ch {channel_idx}'

    # Use full duration
    full_duration = min_samples / sfreq
    time = np.arange(min_samples) / sfreq

    # Get data for first channel
    if include_original:
        data_orig_ch = data_original[channel_idx, :min_samples] * 1e6  # Convert to µV
    data_prep_ch = data_preprocessed[channel_idx, :min_samples] * 1e6
    data_out_ch = data_output[channel_idx, :min_samples] * 1e6

    # Compute correlation
    corr = np.corrcoef(data_prep_ch, data_out_ch)[0, 1]
    mse = np.mean((data_prep_ch - data_out_ch) ** 2)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(20, 4))

    # Plot
    if include_original:
        ax.plot(time, data_orig_ch, 'gray', alpha=0.5, linewidth=0.5, label='Original', linestyle=':')
    ax.plot(time, data_prep_ch, 'b-', alpha=0.7, linewidth=0.6, label='Preprocessed')
    ax.plot(time, data_out_ch, 'r-', alpha=0.7, linewidth=0.6, label='Reconstructed')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude (µV)', fontsize=12)
    ax.set_title(f'FIF File {file_idx}: Full Duration - Channel {ch_name}\nr={corr:.4f}, MSE={mse:.6e}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / f"fif_file{file_idx}_comparison_full_ch{channel_idx}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def compare_plot_pipeline(
    input_dir: str,
    fif_input_dir: str,
    fif_output_dir: str,
    pt_input_dir: str,
    pt_output_dir: str,
    output_dir: str,
    plot_pt: bool = False,
    plot_fif: bool = True,
    num_samples: int = 2,
):
    """
    Generate comparison plots between pipeline input and output.

    Compares preprocessed vs reconstructed files to visually inspect
    model quality. Plots are saved as images to the output_dir.

    Supports comparing both .pt files (epoch-level, before/after model)
    and .fif files (full recording, preprocessed vs reconstructed).

    Args:
        input_dir: Directory containing the original input .fif files.
        fif_input_dir: Directory with preprocessed .fif files (1_fif_filter/).
        fif_output_dir: Directory with reconstructed .fif files (4_fif_output/).
        pt_input_dir: Directory with preprocessed .pt files (2_pt_input/).
        pt_output_dir: Directory with model output .pt files (3_pt_output/).
        output_dir: Directory to save comparison plot images.
        plot_pt: Compare .pt files — shows per-epoch signal comparisons
            between preprocessed input and model output (default: False).
        plot_fif: Compare .fif files — shows full-recording signal overlays
            between preprocessed and reconstructed files (default: True).
        num_samples: Number of files to compare (default: 2).

    Example:
        >>> from zuna import compare_plot_pipeline
        >>> compare_plot_pipeline(
        ...     input_dir="/data/eeg/raw_fif",
        ...     fif_input_dir="/data/eeg/working/1_fif_filter",
        ...     fif_output_dir="/data/eeg/working/4_fif_output",
        ...     pt_input_dir="/data/eeg/working/2_pt_input",
        ...     pt_output_dir="/data/eeg/working/3_pt_output",
        ...     output_dir="/data/eeg/working/FIGURES",
        ...     plot_fif=True,
        ... )
    """
    # Hardcoded defaults (not exposed as flags)
    sample_from_ends = True
    include_original_fif = False
    normalize_for_comparison = False
    # Create output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Get list of files
    fif_original_files = sorted(Path(input_dir).glob("*.fif")) if Path(input_dir).exists() else []
    fif_input_files = sorted(Path(fif_input_dir).glob("*.fif"))
    fif_output_files = sorted(Path(fif_output_dir).glob("*.fif"))
    pt_input_files = sorted(Path(pt_input_dir).glob("*.pt"))
    pt_output_files = sorted(Path(pt_output_dir).glob("*.pt"))

    # Show PT file summary for ALL files
    if len(pt_input_files) > 0 and len(pt_output_files) > 0:

        for pt_input in pt_input_files:
            # Find matching output
            pt_output = None
            for f in pt_output_files:
                if f.name == pt_input.name:
                    pt_output = f
                    break

            if pt_output is None:
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

    # Compare .pt files FIRST (so we get these even if FIF crashes)
    if plot_pt and len(pt_input_files) > 0 and len(pt_output_files) > 0:
        # Sample files (from ends or randomly)
        max_files = min(len(pt_input_files), len(pt_output_files))
        if sample_from_ends:
            # Pick first and last files
            if num_samples == 1:
                sample_indices = [0]
            elif num_samples >= 2:
                sample_indices = [0, max_files - 1]
            else:
                sample_indices = []
        else:
            # Random sampling
            sample_indices = random.sample(range(max_files), min(num_samples, max_files))

        for idx, file_idx in enumerate(sample_indices):
            input_file = pt_input_files[file_idx]
            # Match output file by name
            output_file = None
            for f in pt_output_files:
                if f.name == input_file.name:
                    output_file = f
                    break

            if output_file is None:
                continue

            compare_pt_files(input_file, output_file, output_dir_path, idx + 1)
    else:
        pass

    # Compare .fif files (after PT files, in case this crashes)
    if include_original_fif:
        # 3-line mode: original, preprocessed, reconstructed
        if len(fif_original_files) > 0 and len(fif_input_files) > 0 and len(fif_output_files) > 0:
            # Sample files (from ends or randomly)
            max_files = min(len(fif_original_files), len(fif_input_files), len(fif_output_files))
            if sample_from_ends:
                # Pick first and last files
                if num_samples == 1:
                    sample_indices = [0]
                elif num_samples >= 2:
                    sample_indices = [0, max_files - 1]
                else:
                    sample_indices = []
            else:
                # Random sampling
                sample_indices = random.sample(range(max_files), min(num_samples, max_files))

            for idx, file_idx in enumerate(sample_indices):
                original_file = fif_original_files[file_idx]

                # Match preprocessed file by name (should be same name or in preprocessed subdir)
                preprocessed_file = None
                for f in fif_input_files:
                    if f.stem == original_file.stem or original_file.stem in f.stem:
                        preprocessed_file = f
                        break

                # Match output file by name
                output_file = None
                for f in fif_output_files:
                    if f.stem == original_file.stem or original_file.stem in f.stem:
                        output_file = f
                        break

                if preprocessed_file is None:
                    continue

                if output_file is None:
                    continue

                compare_fif_files(original_file, 
                                  preprocessed_file, 
                                  output_file, 
                                  output_dir_path, 
                                  idx + 1, 
                                  include_original_fif, 
                                  normalize_for_comparison
                )
        else:
            pass
    else:
        # 2-line mode: preprocessed vs reconstructed only
        if plot_fif and len(fif_input_files) > 0 and len(fif_output_files) > 0:
            # Sample files (from ends or randomly)
            max_files = min(len(fif_input_files), len(fif_output_files))
            if sample_from_ends:
                # Pick first and last files
                if num_samples == 1:
                    sample_indices = [0]
                elif num_samples >= 2:
                    sample_indices = [0, max_files - 1]
                else:
                    sample_indices = []
            else:
                # Random sampling
                sample_indices = random.sample(range(max_files), min(num_samples, max_files))

            for idx, file_idx in enumerate(sample_indices):
                preprocessed_file = fif_input_files[file_idx]

                # Match output file by name
                output_file = None
                for f in fif_output_files:
                    if f.stem == preprocessed_file.stem or preprocessed_file.stem in f.stem:
                        output_file = f
                        break

                if output_file is None:
                    continue

                # Pass None for original_file to indicate 2-line mode
                compare_fif_files(None, 
                                  preprocessed_file, 
                                  output_file, 
                                  output_dir_path, 
                                  idx + 1, 
                                  include_original_fif, 
                                  normalize_for_comparison
                )
        else:
            pass
