"""
Utilities for converting between PyTorch PT files and MNE Epochs.
"""

import mne
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def pt_to_mne_epochs(pt_data: Dict) -> mne.Epochs:
    """
    Convert PT file data to MNE Epochs object.

    Args:
        pt_data: Dictionary loaded from PT file with keys:
                 - 'data': list of tensors (n_channels, n_times)
                 - 'channel_positions': list of position tensors
                 - 'labels': tensor of labels
                 - 'metadata': dict with channel_names, sampling_rate, etc.

    Returns:
        MNE Epochs object
    """
    # Extract metadata
    metadata = pt_data['metadata']
    ch_names = metadata['channel_names']
    sfreq = metadata['sampling_rate']
    n_epochs = len(pt_data['data'])

    # Create MNE info
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types='eeg'
    )

    # Create montage from channel positions
    # Use first epoch's positions (they should all be the same)
    positions_3d = pt_data['channel_positions'][0].numpy()

    ch_pos = {ch: pos for ch, pos in zip(ch_names, positions_3d)}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    info.set_montage(montage)

    # Stack data into array (n_epochs, n_channels, n_times)
    epochs_data = torch.stack(pt_data['data']).numpy()

    # Create events array (required by MNE)
    events = np.column_stack([
        np.arange(n_epochs),
        np.zeros(n_epochs, dtype=int),
        pt_data['labels'].numpy()
    ])

    # Create event_id from class_mapping
    class_mapping = metadata['class_mapping']
    event_id = {v: int(k) for k, v in class_mapping.items()}

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


def mne_epochs_to_pt_format(epochs: mne.Epochs, original_pt_data: Dict) -> List[torch.Tensor]:
    """
    Convert MNE Epochs back to PT format (list of tensors).

    Args:
        epochs: MNE Epochs object
        original_pt_data: Original PT data dict (for reference)

    Returns:
        List of tensors, one per epoch
    """
    # Get data as numpy array
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)

    # Convert to list of tensors
    data_list = [torch.from_numpy(data[i]).float() for i in range(len(data))]

    return data_list


def mne_epochs_to_pt_dict(epochs: mne.Epochs, original_pt_data: Dict) -> Dict:
    """
    Convert MNE Epochs back to full PT dict format, preserving all metadata.

    Args:
        epochs: MNE Epochs object
        original_pt_data: Original PT data dict (used to preserve metadata)

    Returns:
        Complete PT data dict with updated data and preserved metadata
    """
    # Convert data
    data_list = mne_epochs_to_pt_format(epochs, original_pt_data)

    # Create new PT dict, preserving all original metadata
    pt_data_new = {
        'data': data_list,
        'labels': original_pt_data['labels'].clone(),
        'channel_positions': original_pt_data['channel_positions'],
        'metadata': original_pt_data['metadata'].copy()
    }

    # Preserve any additional keys from original
    for key in original_pt_data.keys():
        if key not in pt_data_new:
            pt_data_new[key] = original_pt_data[key]

    return pt_data_new


def mark_zero_variance_channels_bad(epochs: mne.Epochs, threshold: float = 1e-10) -> List[str]:
    """
    Mark channels with zero or near-zero variance as bad in MNE Epochs.

    Args:
        epochs: MNE Epochs object (modified in-place)
        threshold: Variance threshold below which channels are marked bad (default: 1e-10)

    Returns:
        List of channel names that were marked as bad
    """
    # Get data as array (n_epochs, n_channels, n_times)
    data = epochs.get_data()

    # Compute variance across all epochs and time points for each channel
    channel_variances = np.var(data, axis=(0, 2))

    # Find channels with variance below threshold
    bad_channel_indices = np.where(channel_variances < threshold)[0]
    bad_channel_names = [epochs.ch_names[i] for i in bad_channel_indices]

    # Mark as bad in epochs
    if len(bad_channel_names) > 0:
        epochs.info['bads'] = list(set(epochs.info.get('bads', []) + bad_channel_names))

    return bad_channel_names


def pt_to_mne_epochs_with_bad_detection(pt_data: Dict, mark_zero_variance: bool = True) -> mne.Epochs:
    """
    Convert PT file to MNE Epochs and optionally mark zero-variance channels as bad.

    Args:
        pt_data: Dictionary loaded from PT file
        mark_zero_variance: If True, automatically mark zero-variance channels as bad

    Returns:
        MNE Epochs object with bad channels marked
    """
    # Convert to MNE
    epochs = pt_to_mne_epochs(pt_data)

    # Mark zero-variance channels as bad
    if mark_zero_variance:
        mark_zero_variance_channels_bad(epochs)

    return epochs


def set_channels_to_zero(pt_data: Dict, percentage: float, seed: Optional[int] = 42,
                         min_channels_keep: int = 3) -> Tuple[Dict, List[str]]:
    """
    Set a percentage of channels to zero in PT file data.

    Args:
        pt_data: Dictionary loaded from PT file
        percentage: Percentage of channels to set to zero (0-100)
                   - 0 = no channels zeroed
                   - 100 = all channels except min_channels_keep zeroed
        seed: Random seed for reproducibility (default: 42, None for no seed)
        min_channels_keep: Minimum number of channels to keep active (default: 3)

    Returns:
        Tuple of (new_pt_data, zeroed_channel_names)
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Get channel info
    ch_names = pt_data['metadata']['channel_names']
    n_channels = len(ch_names)

    # Calculate number of channels to zero
    # 0% = 0 channels, 100% = n_channels - min_channels_keep
    if percentage == 0:
        n_to_zero = 0
    elif percentage == 100:
        n_to_zero = max(0, n_channels - min_channels_keep)
    else:
        n_to_zero = int(np.round(n_channels * percentage / 100.0))
        # Ensure at least min_channels_keep remain non-zero
        max_can_zero = n_channels - min_channels_keep
        if n_to_zero > max_can_zero:
            n_to_zero = max(0, max_can_zero)

    # Select channels to zero
    if n_to_zero == 0:
        zeroed_channels = []
        channel_indices = []
    else:
        channel_indices = np.random.choice(n_channels, size=n_to_zero, replace=False)
        zeroed_channels = [ch_names[i] for i in sorted(channel_indices)]

    # Create new PT data with channels zeroed
    new_data = []
    for epoch_tensor in pt_data['data']:
        epoch_array = epoch_tensor.clone()
        for idx in channel_indices if n_to_zero > 0 else []:
            epoch_array[idx, :] = 0.0
        new_data.append(epoch_array)

    # Create new PT dict with modified data
    pt_data_new = {
        'data': new_data,
        'labels': pt_data['labels'].clone(),
        'channel_positions': pt_data['channel_positions'],
        'metadata': pt_data['metadata'].copy()
    }

    # Preserve any additional keys from original
    for key in pt_data.keys():
        if key not in pt_data_new:
            pt_data_new[key] = pt_data[key]

    # Add metadata about zeroed channels
    if 'zeroed_channels' not in pt_data_new['metadata']:
        pt_data_new['metadata']['zeroed_channels'] = {}
    pt_data_new['metadata']['zeroed_channels'] = {
        'percentage': percentage,
        'n_channels_zeroed': n_to_zero,
        'channel_names': zeroed_channels
    }

    return pt_data_new, zeroed_channels

def interpolate_signals_with_mne(
    signals: List[np.ndarray],
    channel_positions: List[np.ndarray],
    sampling_rate: float = 256.0,
    mark_zero_variance: bool = True,
    verbose: bool = False
) -> List[np.ndarray]:
    """
    Apply MNE interpolation to a list of signals with channel positions.
    Each sample is interpolated individually to handle varying channel counts.

    Args:
        signals: List of numpy arrays, each shape [num_chans, num_times]
        channel_positions: List of numpy arrays, each shape [num_chans, 3]
        sampling_rate: Sampling rate in Hz
        mark_zero_variance: If True, mark zero-variance channels as bad

    Returns:
        List of interpolated numpy arrays, same format as input
    """
    interpolated_signals = []

    # Process each sample individually
    for i, (sig, pos) in enumerate(zip(signals, channel_positions)):
        num_chans = sig.shape[0]

        # Create minimal PT dict for this single sample
        pt_data = {
            'data': [torch.from_numpy(sig).float()],
            'channel_positions': [torch.from_numpy(pos).float()],
            'labels': torch.zeros(1, dtype=torch.long),  # Must be integers for MNE
            'metadata': {
                'channel_names': [f'Ch{j+1}' for j in range(num_chans)],
                'sampling_rate': sampling_rate,
                'class_mapping': {'0': 'event'}  # Need at least one event type for MNE
            }
        }

        # Convert to MNE, mark bad channels, interpolate
        try:
            # Check if positions are valid
            if pos.shape != (num_chans, 3):
                raise ValueError(f"Invalid position shape: {pos.shape}, expected ({num_chans}, 3)")

            # Check if positions have valid values (not all zeros)
            if np.allclose(pos, 0):
                raise ValueError("All channel positions are zero - cannot create montage")

            epochs = pt_to_mne_epochs_with_bad_detection(pt_data, mark_zero_variance=mark_zero_variance)

            # Check if any channels were marked as bad
            if len(epochs.info['bads']) == 0:
                # No bad channels, no need to interpolate
                if verbose: print(f"Sample {i}: No bad channels detected, skipping interpolation")
                interpolated_signals.append(sig)
            else:
                if verbose: print(f"Sample {i}: Interpolating {len(epochs.info['bads'])} bad channels: {epochs.info['bads']}")
                epochs.interpolate_bads(reset_bads=True)

                # Convert back to numpy
                interpolated_data = epochs.get_data()[0]  # [n_channels, n_times]
                interpolated_signals.append(interpolated_data)

        except Exception as e:
            print(f"Warning: MNE interpolation failed for sample {i}: {e}")
            print(f"  Signal shape: {sig.shape}, Position shape: {pos.shape}")
            print(f"  Returning original signal for this sample.")
            interpolated_signals.append(sig)

    return interpolated_signals



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# For subsampling the EGI montage for Localize-MI evals dataset
#
# def egi_montage_subsampling(montage: int, subject_coords: "torch.Tensor"):
#     """
#     Subsample the EGI montage for Localize-MI evals dataset.

#     Args:
#         montage: The montage to subsample
#         subject_coords: The subject coordinates

#     Returns:
#         The dropped indices

#         Developed from: https://github.com/iTCf/mikulan_et_al_2020/blob/1bfed2384b523c8ffcc98b09faad3e94b3e0b138/fx_source_loc.py#L128
#     """
#     if montage not in [16, 32, 64, 128, 256]:
#         raise ValueError("Montage must be one of [16, 32, 64, 128, 256]")

#     subj_pos = (
#         subject_coords.detach().cpu().numpy()
#         if hasattr(subject_coords, "detach")
#         else np.asarray(subject_coords)
#     )
#     if subj_pos.shape != (256, 3):
#         raise ValueError(f"subject_coords must have shape (256,3), got {subj_pos.shape}")

#     subj_names = [f"E{i+1}" for i in range(256)]

#     if montage != 16:
#         montages = (
#             "SUBJECT-256",
#             "GSN-HydroCel-128",
#             "GSN-HydroCel-64_1.0",
#             "GSN-HydroCel-32",
#         )

#         all_monts = {}
#         all_pos = {}
#         all_names = {}

#         all_names["SUBJECT-256"] = subj_names
#         all_pos["SUBJECT-256"] = subj_pos
#         all_monts["SUBJECT-256"] = None

#         for m in montages[1:]:
#             mont = mne.channels.make_standard_montage(m)
#             ch_names = mont.ch_names

#             ch_pos = mont._get_ch_pos()
#             pos = np.array([ch_pos[k] for k in ch_pos.keys()])

#             if m == "GSN-HydroCel-32":
#                 ch_names = ch_names[:-1]
#                 pos = pos[:-1]

#             all_names[m] = ch_names
#             all_pos[m] = pos
#             all_monts[m] = mont

#         if montage == 256:
#             names_256 = all_names["SUBJECT-256"]
#             return {"names": list(names_256), "dists": [0.0] * len(names_256)}

#         # Only compute mapping for requested montage
#         if montage == 32:
#             target_mont_name = "GSN-HydroCel-32"
#         elif montage == 64:
#             target_mont_name = "GSN-HydroCel-64_1.0"
#         elif montage == 128:
#             target_mont_name = "GSN-HydroCel-128"
#         else:
#             raise ValueError(f"Unhandled montage={montage}")

#         subsamp_chs = []
#         subsamp_dists = []
#         used = set()  # enforce uniqueness of selected subject electrodes

#         for c, p in zip(all_names[target_mont_name], all_pos[target_mont_name]):
#             dist_all = np.sqrt(np.sum((all_pos["SUBJECT-256"] - p) ** 2, axis=1))

#             # Legacy special-cases, but now uniqueness-safe
#             if ((target_mont_name == "GSN-HydroCel-128") and (c == "E11")) or (
#                 (target_mont_name == "GSN-HydroCel-64_1.0") and (c == "E8")
#             ):
#                 preferred_name = "E15"
#                 if preferred_name not in used:
#                     used.add(preferred_name)
#                     subsamp_dists.append(float(dist_all[14]))  # index 14 corresponds to E15
#                     subsamp_chs.append(preferred_name)
#                 else:
#                     # fall back to nearest unused
#                     for j in np.argsort(dist_all):
#                         j = int(j)
#                         name_j = all_names["SUBJECT-256"][j]
#                         if name_j not in used:
#                             used.add(name_j)
#                             subsamp_dists.append(float(dist_all[j]))
#                             subsamp_chs.append(name_j)
#                             break
#             else:
#                 # pick nearest unused subject electrode
#                 for j in np.argsort(dist_all):
#                     j = int(j)
#                     name_j = all_names["SUBJECT-256"][j]
#                     if name_j not in used:
#                         used.add(name_j)
#                         subsamp_dists.append(float(dist_all[j]))
#                         subsamp_chs.append(name_j)
#                         break

#         kept = set(subsamp_chs)
#         all_256_names = list(all_names["SUBJECT-256"])
#         dropped_indices = [i for i, ch in enumerate(all_256_names) if ch not in kept]

#     if montage == 16:
#         dropped_indices = make_16_montage(subject_coords)

#     return dropped_indices


# def make_16_montage(subject_coords: "torch.Tensor"):
#     """
#     We don't have a standard 16-channel EGI montage, so create one by:
#     1. Selecting maximum pairwise distance points to get 16 "spread out" channels from the 32-channel montage
#     2. Mapping the size-16 subset of the 32-channel montage to the nearest per-subject 256-channel electrodes
#     """

#     def montage_xyz(montage_name: str):
#         mont = mne.channels.make_standard_montage(montage_name)
#         ch_pos = mont._get_ch_pos()
#         names = list(ch_pos.keys())
#         xyz = np.array([ch_pos[k] for k in names], dtype=float)

#         # Drop reference chans in the 32 montage: keep only "E*"
#         if montage_name == "GSN-HydroCel-32":
#             keep = [i for i, n in enumerate(names) if n.startswith("E")]
#             names = [names[i] for i in keep]
#             xyz = xyz[keep]

#         return names, xyz

#     def nearest_map(source_xyz, target_xyz):
#         diffs = source_xyz[:, None, :] - target_xyz[None, :, :]
#         dists = np.sqrt(np.sum(diffs**2, axis=2))
#         nn_idx = np.argmin(dists, axis=1)
#         nn_dist = dists[np.arange(dists.shape[0]), nn_idx]
#         return nn_idx, nn_dist

#     def farthest_point_sampling(xyz, k, start_idx=None):
#         n = xyz.shape[0]
#         if k > n:
#             raise ValueError(f"k={k} > n={n}")

#         if start_idx is None:
#             start_idx = int(np.argmax(np.sum(xyz**2, axis=1)))

#         selected = [start_idx]
#         d2 = np.sum((xyz - xyz[start_idx])**2, axis=1)

#         for _ in range(1, k):
#             j = int(np.argmax(d2))
#             selected.append(j)
#             d2 = np.minimum(d2, np.sum((xyz - xyz[j])**2, axis=1))

#         return selected

#     xyz256 = (
#         subject_coords.detach().cpu().numpy()
#         if hasattr(subject_coords, "detach")
#         else np.asarray(subject_coords)
#     )
#     if xyz256.shape != (256, 3):
#         raise ValueError(f"subject_coords must have shape (256,3), got {xyz256.shape}")
#     names256 = [f"E{i+1}" for i in range(256)]

#     names32, xyz32 = montage_xyz("GSN-HydroCel-32")

#     k = 16
#     start_idx = None

#     sel32 = farthest_point_sampling(xyz32, k=k, start_idx=start_idx)
#     nn_idx, nn_dist = nearest_map(xyz32[sel32], xyz256)  # selected 32 -> subject 256

#     kept_256 = []
#     kept_d = []
#     for idx, d in zip(nn_idx, nn_dist):
#         idx = int(idx)
#         if idx not in kept_256:
#             kept_256.append(idx)
#             kept_d.append(float(d))

#     kept_256 = sorted(kept_256)
#     kept_names = [names256[i] for i in kept_256]  # retained for parity / debugging
#     dropped_256 = sorted(set(range(len(names256))) - set(kept_256))

#     return dropped_256


# def kept_indices_from_dropped(dropped_indices, n=256):
#     dropped = set(int(i) for i in dropped_indices)
#     return [i for i in range(n) if i not in dropped]

# def print_xyz_stats(name, xyz):
#     mean = xyz.mean(axis=0)
#     std = xyz.std(axis=0)
#     print(
#         f"{name:>12s} | "
#         f"x: {mean[0]: .4f} ± {std[0]: .4f} | "
#         f"y: {mean[1]: .4f} ± {std[1]: .4f} | "
#         f"z: {mean[2]: .4f} ± {std[2]: .4f}"
#     )