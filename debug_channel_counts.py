#!/usr/bin/env python3
"""
Debug script to check channel counts in PT files and understand
why the model filters exactly 50% of epochs.
"""

import torch
import numpy as np
from pathlib import Path

PT_INPUT_DIR = "/data/datasets/bci/dataset_downloads_cw/pip_test/2_pt_input"
PT_OUTPUT_DIR = "/data/datasets/bci/dataset_downloads_cw/pip_test/3_pt_output"

print("="*80)
print("CHANNEL COUNT ANALYSIS")
print("="*80)

# Check all input PT files
pt_input_files = sorted(Path(PT_INPUT_DIR).glob("*.pt"))

for pt_file in pt_input_files:
    print(f"\n{'='*80}")
    print(f"FILE: {pt_file.name}")
    print(f"{'='*80}")

    # Load input
    data_in = torch.load(pt_file, weights_only=False)
    epochs_in = data_in['data']

    # Check channel counts for each epoch in INPUT
    print("\nINPUT PT file:")
    channel_counts = []
    for i, epoch in enumerate(epochs_in):
        if epoch is not None:
            arr = epoch.numpy() if isinstance(epoch, torch.Tensor) else epoch
            n_channels = arr.shape[0]
            channel_counts.append(n_channels)
        else:
            channel_counts.append(0)

    # Count distribution
    from collections import Counter
    count_dist = Counter(channel_counts)
    print(f"  Total epochs: {len(epochs_in)}")
    print(f"  Channel count distribution:")
    for n_chan, count in sorted(count_dist.items()):
        if n_chan == 0:
            print(f"    None epochs: {count}")
        else:
            print(f"    {n_chan} channels: {count} epochs ({100*count/len(epochs_in):.0f}%)")

    # Check if there's any variation
    non_zero_counts = [c for c in channel_counts if c > 0]
    if len(set(non_zero_counts)) > 1:
        print(f"  ⚠️  VARIATION DETECTED: Not all epochs have the same channel count!")
        # Show which epochs have different counts
        unique_counts = sorted(set(non_zero_counts))
        for n in unique_counts:
            indices = [i for i, c in enumerate(channel_counts) if c == n]
            print(f"    {n} channels at indices: {indices[:20]}{'...' if len(indices) > 20 else ''}")

    # Load output and check which were filtered
    pt_output = Path(PT_OUTPUT_DIR) / pt_file.name
    if pt_output.exists():
        data_out = torch.load(pt_output, weights_only=False)
        epochs_out = data_out['data']

        print("\nOUTPUT PT file (after model):")
        valid_indices = [i for i, e in enumerate(epochs_out) if e is not None]
        none_indices = [i for i, e in enumerate(epochs_out) if e is None]

        print(f"  Valid epochs: {len(valid_indices)} ({100*len(valid_indices)/len(epochs_out):.0f}%)")
        print(f"  None epochs:  {len(none_indices)} ({100*len(none_indices)/len(epochs_out):.0f}%)")

        # Check if there's a pattern between input channel counts and output None
        if len(non_zero_counts) > 0 and len(set(non_zero_counts)) > 1:
            print("\nCORRELATION: Input channel count vs Output None:")
            for n in unique_counts:
                indices_with_n_channels = [i for i, c in enumerate(channel_counts) if c == n]
                none_from_this_count = [i for i in indices_with_n_channels if i in none_indices]
                print(f"  {n} channels: {len(none_from_this_count)}/{len(indices_with_n_channels)} became None ({100*len(none_from_this_count)/len(indices_with_n_channels):.0f}%)")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nIf all input epochs have the same channel count (32), but 50% become None,")
print("then the model's chan_num_filter is probably set to a DIFFERENT value (not 32).")
print("\nIf input epochs have VARYING channel counts, then the model is filtering")
print("based on that variation.")
