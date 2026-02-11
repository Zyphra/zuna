#!/usr/bin/env python3
"""
Analyze PT files before and after model inference.

This script:
1. Checks all PT input files (preprocessing output)
2. Checks all PT output files (model output)
3. Compares them to see which epochs are None or zero
4. Maps PT files to their original FIF files
"""

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

# Configuration
PT_INPUT_DIR = "/data/datasets/bci/dataset_downloads_cw/pip_test/2_pt_input"
PT_OUTPUT_DIR = "/data/datasets/bci/dataset_downloads_cw/pip_test/3_pt_output"

def analyze_pt_file(pt_path: Path, is_output: bool = False):
    """Analyze a single PT file and return statistics."""
    data = torch.load(pt_path, weights_only=False)
    metadata = data.get('metadata', {})
    pt_data = data.get('data', [])

    n_epochs = len(pt_data)
    none_epochs = 0
    zero_epochs = 0
    valid_epochs = 0

    for epoch in pt_data:
        if epoch is None:
            none_epochs += 1
        elif isinstance(epoch, (torch.Tensor, np.ndarray)):
            epoch_array = epoch.numpy() if isinstance(epoch, torch.Tensor) else epoch
            if np.all(epoch_array == 0):
                zero_epochs += 1
            else:
                valid_epochs += 1

    return {
        'filename': pt_path.name,
        'original_filename': metadata.get('original_filename', 'UNKNOWN'),
        'n_epochs': n_epochs,
        'none_epochs': none_epochs,
        'zero_epochs': zero_epochs,
        'valid_epochs': valid_epochs,
        'metadata': metadata
    }

def main():
    print("="*80)
    print("PT FILE ANALYSIS")
    print("="*80)

    # Analyze input PT files
    print("\n" + "="*80)
    print("PT INPUT FILES (Preprocessing Output)")
    print("="*80)

    input_dir = Path(PT_INPUT_DIR)
    input_files = sorted(input_dir.glob("*.pt"))

    if len(input_files) == 0:
        print("  No PT input files found!")
    else:
        input_stats = []
        for pt_file in input_files:
            stats = analyze_pt_file(pt_file, is_output=False)
            input_stats.append(stats)

            print(f"\n{stats['filename']}")
            print(f"  Original FIF: {stats['original_filename']}")
            print(f"  Total epochs: {stats['n_epochs']}")
            print(f"  Valid epochs: {stats['valid_epochs']}")
            print(f"  None epochs:  {stats['none_epochs']}")
            print(f"  Zero epochs:  {stats['zero_epochs']}")
            if stats['none_epochs'] > 0 or stats['zero_epochs'] > 0:
                print(f"  ⚠️  WARNING: Contains None or zero epochs!")

    # Analyze output PT files
    print("\n" + "="*80)
    print("PT OUTPUT FILES (Model Output)")
    print("="*80)

    output_dir = Path(PT_OUTPUT_DIR)
    output_files = sorted(output_dir.glob("*.pt"))

    if len(output_files) == 0:
        print("  No PT output files found!")
    else:
        output_stats = []
        for pt_file in output_files:
            stats = analyze_pt_file(pt_file, is_output=True)
            output_stats.append(stats)

            print(f"\n{stats['filename']}")
            print(f"  Original FIF: {stats['original_filename']}")
            print(f"  Total epochs: {stats['n_epochs']}")
            print(f"  Valid epochs: {stats['valid_epochs']}")
            print(f"  None epochs:  {stats['none_epochs']}")
            print(f"  Zero epochs:  {stats['zero_epochs']}")

            if stats['none_epochs'] > 0:
                pct = 100 * stats['none_epochs'] / stats['n_epochs']
                print(f"  ⚠️  {pct:.1f}% of epochs are None (filtered by model)")
            if stats['zero_epochs'] > 0:
                pct = 100 * stats['zero_epochs'] / stats['n_epochs']
                print(f"  ⚠️  {pct:.1f}% of epochs are all zeros")

    # Comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    if len(input_files) > 0 and len(output_files) > 0:
        # Group by original filename
        input_by_orig = defaultdict(list)
        output_by_orig = defaultdict(list)

        for stats in input_stats:
            input_by_orig[stats['original_filename']].append(stats)

        for stats in output_stats:
            output_by_orig[stats['original_filename']].append(stats)

        all_originals = set(input_by_orig.keys()) | set(output_by_orig.keys())

        for orig_filename in sorted(all_originals):
            print(f"\nOriginal FIF: {orig_filename}")

            # Input summary
            input_list = input_by_orig.get(orig_filename, [])
            if input_list:
                total_input_epochs = sum(s['n_epochs'] for s in input_list)
                total_input_valid = sum(s['valid_epochs'] for s in input_list)
                print(f"  Input:  {len(input_list)} PT files, {total_input_epochs} epochs total, {total_input_valid} valid")
            else:
                print(f"  Input:  NO FILES")

            # Output summary
            output_list = output_by_orig.get(orig_filename, [])
            if output_list:
                total_output_epochs = sum(s['n_epochs'] for s in output_list)
                total_output_valid = sum(s['valid_epochs'] for s in output_list)
                total_output_none = sum(s['none_epochs'] for s in output_list)
                total_output_zero = sum(s['zero_epochs'] for s in output_list)

                print(f"  Output: {len(output_list)} PT files, {total_output_epochs} epochs total")
                print(f"          {total_output_valid} valid, {total_output_none} None, {total_output_zero} zero")

                if total_output_none > 0:
                    pct = 100 * total_output_none / total_output_epochs
                    print(f"          ⚠️  {pct:.1f}% filtered by model (channel count mismatch)")
            else:
                print(f"  Output: NO FILES")

    print("\n" + "="*80)
    print("NOTES")
    print("="*80)
    print("- None epochs: Filtered out by the model (usually due to channel count mismatch)")
    print("- Zero epochs: All values are exactly zero (should not happen in input)")
    print("- Valid epochs: Have non-zero data and passed through the model")
    print("="*80)

if __name__ == "__main__":
    main()
