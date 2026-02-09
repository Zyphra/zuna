import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load input and output files
data_input = "data/2_pt_input/ds000000_000000_000001_d00_00003_32_1280.pt"
data_output = "data/3_pt_output/ds000000_000000_000001_d00_00003_32_1280.pt"

di = torch.load(data_input, weights_only=False)
do = torch.load(data_output, weights_only=False)

print("=" * 80)
print("FILE STRUCTURE COMPARISON")
print("=" * 80)
print(f"Input keys:  {di.keys()}")
print(f"Output keys: {do.keys()}")
print(f"Input has {len(di['data'])} samples")
print(f"Output has {len(do['data'])} samples")
print()

# Compare metadata
print("=" * 80)
print("METADATA COMPARISON")
print("=" * 80)
print("Metadata matches:", di['metadata'] == do['metadata'])
if di['metadata'] != do['metadata']:
    print("Differences found in metadata")
print()

# Numeric comparison for first 3 samples
print("=" * 80)
print("NUMERIC COMPARISON (First 3 samples)")
print("=" * 80)

num_samples_to_compare = min(3, len(di['data']))

for i in range(num_samples_to_compare):
    print(f"\n--- Sample {i} ---")

    # Original input vs original in output (should match perfectly)
    orig_input = di['data'][i]  # Original from input file
    orig_output = do['data_original'][i]  # Original saved in output file
    recon_output = do['data'][i]  # Reconstructed from model

    # Check if samples exist
    if orig_output is None:
        print(f"  WARNING: Sample {i} original data is None (not processed)")
        continue
    if recon_output is None:
        print(f"  WARNING: Sample {i} reconstruction is None (not processed)")
        continue

    # Convert to numpy for easier comparison
    if isinstance(orig_input, torch.Tensor):
        orig_input = orig_input.numpy()
    if isinstance(orig_output, torch.Tensor):
        orig_output = orig_output.numpy()
    if isinstance(recon_output, torch.Tensor):
        recon_output = recon_output.numpy()

    print(f"  Original shape: {orig_input.shape}")
    print(f"  Reconstructed shape: {recon_output.shape}")

    # IMPORTANT: Check if normalization needs to be reversed
    # The pipeline divides by 10.0, so if the saved data is also divided, we need to account for this
    orig_input_range = (orig_input.min(), orig_input.max())
    orig_output_range = (orig_output.min(), orig_output.max())
    recon_output_range = (recon_output.min(), recon_output.max())
    print(f"  Input range: [{orig_input_range[0]:.3f}, {orig_input_range[1]:.3f}]")
    print(f"  Output orig range: [{orig_output_range[0]:.3f}, {orig_output_range[1]:.3f}]")
    print(f"  Recon range: [{recon_output_range[0]:.3f}, {recon_output_range[1]:.3f}]")

    # Check if original data was preserved correctly
    orig_diff = np.abs(orig_input - orig_output).max()
    print(f"  Max diff (input vs output original): {orig_diff:.2e}")

    # Compare reconstruction to original
    mse = np.mean((orig_input - recon_output) ** 2)
    mae = np.mean(np.abs(orig_input - recon_output))

    # Compute correlation per channel
    correlations = []
    for ch in range(orig_input.shape[0]):
        corr = np.corrcoef(orig_input[ch], recon_output[ch])[0, 1]
        correlations.append(corr)
    mean_corr = np.mean(correlations)

    print(f"  MSE (original vs reconstruction): {mse:.6f}")
    print(f"  MAE (original vs reconstruction): {mae:.6f}")
    print(f"  Mean correlation per channel: {mean_corr:.4f}")
    print(f"  Correlation range: [{min(correlations):.4f}, {max(correlations):.4f}]")

print()
print("=" * 80)
print("PLOTTING COMPARISON")
print("=" * 80)

# Create comparison plots
output_dir = Path("data/4_comparison_plots")
output_dir.mkdir(parents=True, exist_ok=True)

for sample_idx in range(num_samples_to_compare):
    orig_input = di['data'][sample_idx]
    recon_output = do['data'][sample_idx]

    if isinstance(orig_input, torch.Tensor):
        orig_input = orig_input.numpy()
    if isinstance(recon_output, torch.Tensor):
        recon_output = recon_output.numpy()

    num_channels, num_timepoints = orig_input.shape

    # Plot all channels
    fig, axes = plt.subplots(num_channels, 1, figsize=(20, 2 * num_channels))
    if num_channels == 1:
        axes = [axes]

    for ch in range(num_channels):
        ax = axes[ch]
        time = np.arange(num_timepoints) / 256  # Assuming 256 Hz sampling rate

        ax.plot(time, orig_input[ch], 'b-', alpha=0.7, linewidth=0.8, label='Original')
        ax.plot(time, recon_output[ch], 'r-', alpha=0.7, linewidth=0.8, label='Reconstructed')

        # Compute correlation for this channel
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

    plt.suptitle(f'Sample {sample_idx}: Original vs Reconstructed', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f"sample_{sample_idx}_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")

print("\nComparison complete!")
print(f"Plots saved to: {output_dir}")
