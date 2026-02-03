#!/usr/bin/env python3
"""
Load PT files from the preprocessing pipeline and visualize them with MNE.

Usage:
    python 7_pt_to_mne_vis.py <path_to_pt_file> [--output_dir 7_pt_to_mne_vis_figures]

Example:
    python 7_pt_to_mne_vis.py /data/datasets/bci/pt3d/faced/dataset_000000_000001_d05_00064_89_1280.pt
    python 7_pt_to_mne_vis.py input.pt --output_dir my_plots
"""

import argparse
import os
from pathlib import Path
import torch
import numpy as np
import mne
import matplotlib.pyplot as plt

mne.set_log_level('ERROR')
plt.ioff()


def pt_to_mne_epochs(pt_data):
    """
    Convert PT file from pipeline format to MNE Epochs object.

    Pipeline PT format:
        - 'data': list of tensors (n_channels, n_times)
        - 'channel_positions': list of position tensors (n_channels, 3)
        - 'metadata': dict with 'resampled_sampling_rate', 'channel_names_10_20', etc.
    """
    metadata = pt_data['metadata']

    # Extract channel names - try different keys
    if 'channel_names_10_20' in metadata:
        ch_names = metadata['channel_names_10_20']
    elif 'channel_names' in metadata:
        ch_names = metadata['channel_names']
    else:
        # Fallback: create generic names based on first epoch
        n_channels = pt_data['data'][0].shape[0]
        ch_names = [f'Ch{i+1}' for i in range(n_channels)]

    # Extract sampling rate
    if 'resampled_sampling_rate' in metadata:
        sfreq = metadata['resampled_sampling_rate']
    elif 'sampling_rate' in metadata:
        sfreq = metadata['sampling_rate']
    else:
        sfreq = 256.0  # Default

    n_epochs = len(pt_data['data'])

    # Get actual channel count from first epoch
    actual_n_channels = pt_data['data'][0].shape[0]

    # Truncate channel names if there are more names than channels
    if len(ch_names) > actual_n_channels:
        print(f"⚠️  Warning: {len(ch_names)} channel names but only {actual_n_channels} channels in data")
        print(f"   Using first {actual_n_channels} channel names")
        ch_names = ch_names[:actual_n_channels]
    elif len(ch_names) < actual_n_channels:
        print(f"⚠️  Warning: Only {len(ch_names)} channel names but {actual_n_channels} channels in data")
        print(f"   Adding generic names for missing channels")
        ch_names = ch_names + [f'Ch{i+1}' for i in range(len(ch_names), actual_n_channels)]

    # Create MNE info
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types='eeg'
    )

    # Create montage from channel positions (use first epoch)
    try:
        positions_3d = pt_data['channel_positions'][0].numpy()

        # Match positions to actual channels
        if positions_3d.shape[0] > actual_n_channels:
            positions_3d = positions_3d[:actual_n_channels]

        ch_pos = {ch: pos for ch, pos in zip(ch_names, positions_3d)}
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
        info.set_montage(montage)
        print(f"✓ Created montage with {len(ch_pos)} channels")
    except Exception as e:
        print(f"⚠️  Warning: Could not create montage: {e}")
        print("   Continuing without spatial information")

    # Stack data into array (n_epochs, n_channels, n_times)
    epochs_data = torch.stack(pt_data['data']).numpy()

    # Create dummy events (pipeline data doesn't have labels)
    events = np.column_stack([
        np.arange(n_epochs) * 100,  # Sample indices (spaced out)
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int)  # All same event type
    ])

    event_id = {'epoch': 1}

    # Create Epochs object
    epochs = mne.EpochsArray(
        epochs_data,
        info,
        events=events,
        event_id=event_id,
        tmin=0,
        verbose=False
    )

    return epochs


def visualize_pt_file(pt_path, output_dir='7_pt_to_mne_vis_figures'):
    """
    Load PT file and create visualizations.

    Args:
        pt_path: Path to PT file
        output_dir: Directory to save plots
    """
    print(f"\n{'='*80}")
    print(f"Loading: {pt_path}")
    print(f"{'='*80}\n")

    # Load PT file
    pt_data = torch.load(pt_path)

    # Print info
    n_epochs = len(pt_data['data'])
    n_channels = pt_data['data'][0].shape[0]
    n_times = pt_data['data'][0].shape[1]

    print(f"📊 PT File Contents:")
    print(f"   Epochs: {n_epochs}")
    print(f"   Channels: {n_channels}")
    print(f"   Samples per epoch: {n_times}")

    if 'metadata' in pt_data:
        metadata = pt_data['metadata']
        if 'resampled_sampling_rate' in metadata:
            sfreq = metadata['resampled_sampling_rate']
            print(f"   Sampling rate: {sfreq} Hz")
            print(f"   Duration per epoch: {n_times/sfreq:.2f} seconds")

        if 'dataset_info' in metadata:
            dataset_info = metadata['dataset_info']
            print(f"\n📁 Dataset Info:")
            for key, value in dataset_info.items():
                if key != 'source_files':  # Skip long list
                    print(f"   {key}: {value}")

        if 'processing_stats' in metadata:
            stats = metadata['processing_stats']
            print(f"\n📈 Processing Stats:")
            if 'n_epochs' in stats:
                print(f"   Total epochs: {stats['n_epochs']}")
            if 'avg_channels_per_epoch' in stats:
                print(f"   Avg channels per epoch: {stats['avg_channels_per_epoch']:.1f}")

    print(f"\n{'='*80}")
    print("Converting to MNE Epochs...")
    print(f"{'='*80}\n")

    # Convert to MNE
    epochs = pt_to_mne_epochs(pt_data)

    print(f"✓ Created MNE Epochs object")
    print(f"   Channels: {len(epochs.ch_names)}")
    print(f"   Epochs: {len(epochs)}")
    print(f"   Sampling rate: {epochs.info['sfreq']} Hz")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate base filename from PT filename
    base_name = Path(pt_path).stem

    print(f"\n{'='*80}")
    print("Generating visualizations...")
    print(f"{'='*80}\n")

    # Plot 1: PSD
    print("📊 Plotting PSD...")
    try:
        fig_psd = epochs.compute_psd(fmax=50).plot(show=False, average=True)
        psd_path = os.path.join(output_dir, f'{base_name}_psd.png')
        fig_psd.savefig(psd_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {psd_path}")
        plt.close(fig_psd)
    except Exception as e:
        print(f"   ✗ PSD plot failed: {e}")

    # Plot 2: PSD topomap (if montage exists)
    if epochs.get_montage() is not None:
        print("🗺️  Plotting PSD topomap...")
        try:
            fig_topo = epochs.compute_psd().plot_topomap(show=False)
            topo_path = os.path.join(output_dir, f'{base_name}_psd_topomap.png')
            fig_topo.savefig(topo_path, dpi=150, bbox_inches='tight')
            print(f"   ✓ Saved: {topo_path}")
            plt.close(fig_topo)
        except Exception as e:
            print(f"   ⚠️  Topomap plot failed: {e}")

    # Plot 3: Raw data traces (first few epochs)
    print("📈 Plotting raw data traces...")
    try:
        n_epochs_to_plot = min(5, len(epochs))
        fig_raw = epochs[:n_epochs_to_plot].plot(
            n_channels=min(30, len(epochs.ch_names)),
            scalings='auto',
            show=False
        )
        raw_path = os.path.join(output_dir, f'{base_name}_traces.png')
        fig_raw.savefig(raw_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {raw_path} (first {n_epochs_to_plot} epochs)")
        plt.close(fig_raw)
    except Exception as e:
        print(f"   ✗ Traces plot failed: {e}")

    # Plot 4: Image plot (all epochs as heatmap)
    print("🔥 Plotting epoch heatmap...")
    try:
        fig_image = epochs.plot_image(
            picks=[0],  # Just first channel
            show=False
        )
        image_path = os.path.join(output_dir, f'{base_name}_heatmap.png')
        fig_image[0].savefig(image_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {image_path}")
        plt.close(fig_image[0])
    except Exception as e:
        print(f"   ⚠️  Heatmap plot failed: {e}")

    print(f"\n{'='*80}")
    print(f"✅ Visualization complete!")
    print(f"   Output directory: {output_dir}")
    print(f"{'='*80}\n")

    return epochs


def main():
    parser = argparse.ArgumentParser(
        description='Visualize PT files from preprocessing pipeline using MNE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 7_pt_to_mne_vis.py input.pt
  python 7_pt_to_mne_vis.py /data/datasets/bci/pt3d/faced/dataset_000000_000001_d05_00064_89_1280.pt --output_dir my_plots
        """
    )
    parser.add_argument('pt_file', type=str, help='Path to PT file')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='Output directory for plots (default: figures)')

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.pt_file):
        print(f"❌ Error: File not found: {args.pt_file}")
        return 1

    # Visualize
    try:
        epochs = visualize_pt_file(args.pt_file, args.output_dir)
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
