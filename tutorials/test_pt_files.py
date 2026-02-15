#!/usr/bin/env python3
"""
Simple test script to compare input PT files (preprocessed) with output PT files (after model).
Checks: SD comparison, NaN values, and metadata matching.
"""

import torch
import numpy as np
from pathlib import Path
import random

# Paths
PT_INPUT_DIR = Path("/data/datasets/bci/dataset_downloads_cw/pip_test/2_pt_input")
PT_OUTPUT_DIR = Path("/data/datasets/bci/dataset_downloads_cw/pip_test/3_pt_output")

# How many epochs to sample for correlation check
NUM_CORRELATION_SAMPLES = 30

def load_pt_file(filepath):
    """Load a PT file and return data and metadata."""
    # weights_only=False is needed for numpy arrays in PT files
    data = torch.load(filepath, map_location='cpu', weights_only=False)
    return data

def check_for_nans(data):
    """Check if tensor has NaN values."""
    if torch.is_tensor(data):
        return torch.isnan(data).any().item()
    elif isinstance(data, np.ndarray):
        return np.isnan(data).any()
    return False

def get_std(data):
    """Get standard deviation of tensor."""
    if torch.is_tensor(data):
        return data.std().item()
    elif isinstance(data, np.ndarray):
        return data.std()
    return None

def compute_correlation(epoch1, epoch2):
    """
    Compute correlation between two epochs.
    Returns correlation coefficient (averaged across channels).
    """
    if epoch1 is None or epoch2 is None:
        return None

    # Convert to numpy if needed
    if torch.is_tensor(epoch1):
        epoch1 = epoch1.cpu().numpy()
    if torch.is_tensor(epoch2):
        epoch2 = epoch2.cpu().numpy()

    # Ensure same shape
    if epoch1.shape != epoch2.shape:
        return None

    # Compute correlation for each channel
    n_channels = epoch1.shape[0]
    correlations = []

    for ch in range(n_channels):
        signal1 = epoch1[ch, :]
        signal2 = epoch2[ch, :]

        # Skip if either has zero variance
        if signal1.std() < 1e-10 or signal2.std() < 1e-10:
            continue

        # Compute Pearson correlation
        corr = np.corrcoef(signal1, signal2)[0, 1]
        correlations.append(corr)

    if correlations:
        return np.mean(correlations)
    return None

def compare_metadata(meta1, meta2):
    """Compare two metadata dictionaries."""
    if meta1 is None and meta2 is None:
        return True, []
    if meta1 is None or meta2 is None:
        return False, ["One metadata is None"]

    differences = []

    # Check keys
    keys1 = set(meta1.keys())
    keys2 = set(meta2.keys())

    if keys1 != keys2:
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        if only_in_1:
            differences.append(f"Keys only in input: {only_in_1}")
        if only_in_2:
            differences.append(f"Keys only in output: {only_in_2}")

    # Check values for common keys
    for key in keys1 & keys2:
        v1, v2 = meta1[key], meta2[key]
        if v1 != v2:
            differences.append(f"  {key}: {v1} → {v2}")

    return len(differences) == 0, differences

def main():
    # Get all input PT files
    input_files = sorted(PT_INPUT_DIR.glob("*.pt"))

    if not input_files:
        print(f"❌ No PT files found in {PT_INPUT_DIR}")
        return

    print("="*80)
    print("PT FILE COMPARISON TEST")
    print("="*80)
    print(f"Input dir:  {PT_INPUT_DIR}")
    print(f"Output dir: {PT_OUTPUT_DIR}")
    print(f"Found {len(input_files)} input PT files\n")

    results = {
        'total': 0,
        'output_missing': 0,
        'has_nans': 0,
        'metadata_mismatch': 0,
        'std_issues': 0
    }

    # Store all valid epoch pairs for correlation analysis
    all_epoch_pairs = []  # List of (filename, epoch_idx, input_epoch, output_epoch)

    for input_file in input_files:
        results['total'] += 1
        output_file = PT_OUTPUT_DIR / input_file.name

        print("="*80)
        print(f"File: {input_file.name}")
        print("="*80)

        # Check if output file exists
        if not output_file.exists():
            print(f"❌ Output file not found!")
            results['output_missing'] += 1
            continue

        # Load files
        try:
            input_data = load_pt_file(input_file)
            output_data = load_pt_file(output_file)
        except Exception as e:
            print(f"❌ Error loading files: {e}")
            continue

        # Extract epochs and metadata
        input_epochs = input_data.get('data', None)
        output_epochs = output_data.get('data', None)
        input_meta = input_data.get('metadata', None)
        output_meta = output_data.get('metadata', None)

        # 1. Check standard deviations
        print("\n1️⃣  STANDARD DEVIATION COMPARISON:")

        if input_epochs is not None and output_epochs is not None:
            # Count None epochs
            input_none_count = sum(1 for e in input_epochs if e is None)
            output_none_count = sum(1 for e in output_epochs if e is None)

            print(f"   Input:  {len(input_epochs)} epochs ({input_none_count} None)")
            print(f"   Output: {len(output_epochs)} epochs ({output_none_count} None)")

            # Get valid epochs
            input_valid = [e for e in input_epochs if e is not None]
            output_valid = [e for e in output_epochs if e is not None]

            if input_valid and output_valid:
                # Concatenate all valid epochs
                input_all = torch.cat(input_valid, dim=0) if torch.is_tensor(input_valid[0]) else np.concatenate(input_valid, axis=0)
                output_all = torch.cat(output_valid, dim=0) if torch.is_tensor(output_valid[0]) else np.concatenate(output_valid, axis=0)

                input_std = get_std(input_all)
                output_std = get_std(output_all)

                print(f"   Input STD:  {input_std:.6e}")
                print(f"   Output STD: {output_std:.6e}")

                if input_std and output_std:
                    ratio = output_std / input_std
                    print(f"   Ratio (output/input): {ratio:.4f}")

                    if abs(ratio - 1.0) > 0.5:  # More than 50% difference
                        print(f"   ⚠️  Large STD difference!")
                        results['std_issues'] += 1
                    else:
                        print(f"   ✓ STD similar")
            else:
                print(f"   ⚠️  No valid epochs for comparison")
                results['std_issues'] += 1
        else:
            print(f"   ❌ Missing epoch data")
            results['std_issues'] += 1

        # 2. Check for NaN values in output
        print("\n2️⃣  NaN CHECK (output):")

        has_nan = False
        if output_valid:
            for i, epoch in enumerate(output_valid):
                if check_for_nans(epoch):
                    print(f"   ❌ NaN found in epoch {i}")
                    has_nan = True

        if has_nan:
            results['has_nans'] += 1
            print(f"   ❌ File has NaN values!")
        else:
            print(f"   ✓ No NaN values found")

        # 3. Compare metadata
        print("\n3️⃣  METADATA COMPARISON:")

        meta_match, differences = compare_metadata(input_meta, output_meta)

        if meta_match:
            print(f"   ✓ Metadata matches")
        else:
            print(f"   ❌ Metadata differences found:")
            for diff in differences:
                print(f"      {diff}")
            results['metadata_mismatch'] += 1

        # 4. Store epoch pairs for correlation analysis
        if input_epochs is not None and output_epochs is not None:
            # Match up valid epochs by index
            for i in range(min(len(input_epochs), len(output_epochs))):
                if input_epochs[i] is not None and output_epochs[i] is not None:
                    all_epoch_pairs.append((input_file.name, i, input_epochs[i], output_epochs[i]))

        print()

    # Correlation analysis across all files
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS (Sampling Random Epochs)")
    print("="*80)

    if all_epoch_pairs:
        # Sample random epochs for correlation check
        num_samples = min(NUM_CORRELATION_SAMPLES, len(all_epoch_pairs))
        sampled_pairs = random.sample(all_epoch_pairs, num_samples)

        print(f"\nTesting {num_samples} randomly sampled epochs across all files...")
        print(f"(Correlation should be very high if epochs are in correct positions)\n")

        correlations = []
        low_corr_epochs = []

        for filename, epoch_idx, input_epoch, output_epoch in sampled_pairs:
            corr = compute_correlation(input_epoch, output_epoch)

            if corr is not None:
                correlations.append(corr)

                # Flag low correlations (< 0.8)
                if corr < 0.8:
                    low_corr_epochs.append((filename, epoch_idx, corr))

        if correlations:
            mean_corr = np.mean(correlations)
            min_corr = np.min(correlations)
            max_corr = np.max(correlations)

            print(f"Correlation Statistics:")
            print(f"  Mean:  {mean_corr:.4f}")
            print(f"  Range: [{min_corr:.4f}, {max_corr:.4f}]")
            print(f"  Samples tested: {len(correlations)}")

            if mean_corr > 0.95:
                print(f"\n  ✓ Excellent correlation! Epochs are in correct positions.")
            elif mean_corr > 0.8:
                print(f"\n  ✓ Good correlation. Epochs likely in correct positions.")
            elif mean_corr > 0.5:
                print(f"\n  ⚠️  Moderate correlation. Check if epochs are misaligned.")
            else:
                print(f"\n  ❌ Low correlation! Epochs may be in wrong positions!")

            if low_corr_epochs:
                print(f"\n  Low correlation epochs (< 0.8):")
                for fname, idx, corr in low_corr_epochs[:10]:  # Show first 10
                    print(f"    {fname}, epoch {idx}: {corr:.4f}")
                if len(low_corr_epochs) > 10:
                    print(f"    ... and {len(low_corr_epochs) - 10} more")
        else:
            print(f"  ❌ Could not compute correlations")
    else:
        print(f"\n  ⚠️  No valid epoch pairs found for correlation analysis")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total files checked: {results['total']}")
    print(f"  ✓ All checks passed: {results['total'] - results['output_missing'] - results['has_nans'] - results['metadata_mismatch'] - results['std_issues']}")
    print(f"  ❌ Output file missing: {results['output_missing']}")
    print(f"  ❌ Has NaN values: {results['has_nans']}")
    print(f"  ❌ Metadata mismatch: {results['metadata_mismatch']}")
    print(f"  ⚠️  STD issues: {results['std_issues']}")
    print("="*80)

if __name__ == "__main__":
    main()
