"""
Channel upsampling utilities for EEG data.
"""
import numpy as np
import mne
from typing import List, Tuple, Optional


def upsample_channels(
    epochs_list: List[np.ndarray],
    positions_list: List[np.ndarray],
    channel_names: List[str],
    target_n_channels: int,
    montage_source: str = 'standard_1005'
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Upsample epochs to target number of channels using greedy selection from standard montage.

    New channels are added with all values set to 0, signaling the model to interpolate them.
    Uses greedy distance-based selection to maximize spatial coverage.

    Algorithm:
    1. Load standard montage (default: standard_1005, ~340 channels)
    2. Find candidate channels: in montage but NOT in current data
    3. For each candidate, compute minimum distance to existing channels
    4. Sort candidates by distance (furthest first = best coverage)
    5. Select top N candidates where N = target_n_channels - current_n_channels
    6. Add new channels with all values = 0
    7. Return upsampled data with new positions and names

    Parameters
    ----------
    epochs_list : list of np.ndarray
        List of epoch arrays, each (n_channels, n_times)
    positions_list : list of np.ndarray
        List of 3D channel position arrays, each (n_channels, 3)
    channel_names : list of str
        Current channel names
    target_n_channels : int
        Target number of channels after upsampling
    montage_source : str, optional
        MNE montage name to use for new channel positions
        Default: 'standard_1005' (densest standard montage)

    Returns
    -------
    upsampled_epochs : list of np.ndarray
        Upsampled epoch arrays, each (target_n_channels, n_times)
    upsampled_positions : list of np.ndarray
        Upsampled position arrays, each (target_n_channels, 3)
    upsampled_names : list of str
        Channel names after upsampling

    Examples
    --------
    >>> # Upsample from 64 to 128 channels
    >>> epochs_up, positions_up, names_up = upsample_channels(
    ...     epochs_list, positions_list, channel_names, target_n_channels=128
    ... )
    >>> print(f"Upsampled from {len(channel_names)} to {len(names_up)} channels")
    """
    if len(epochs_list) == 0:
        return epochs_list, positions_list, channel_names

    current_n_channels = len(channel_names)

    if current_n_channels >= target_n_channels:
        raise ValueError(
            f"Current channel count ({current_n_channels}) >= target ({target_n_channels}). "
            "No upsampling needed."
        )

    n_channels_to_add = target_n_channels - current_n_channels

    # Load source montage
    try:
        montage = mne.channels.make_standard_montage(montage_source)
    except Exception as e:
        raise ValueError(f"Failed to load montage '{montage_source}': {e}")

    # Get all positions from montage
    montage_pos = montage.get_positions()['ch_pos']

    # Normalize channel names for comparison
    current_names_normalized = {name.lower(): name for name in channel_names}
    montage_names_normalized = {name.lower(): name for name in montage_pos.keys()}

    # Find candidate channels (in montage but not in current data)
    candidate_names = []
    candidate_positions = []

    for norm_name, orig_name in montage_names_normalized.items():
        if norm_name not in current_names_normalized:
            pos = montage_pos[orig_name]
            pos_array = np.array([pos[0], pos[1], pos[2]])
            # Skip if position is all zeros
            if not np.allclose(pos_array, [0.0, 0.0, 0.0]):
                candidate_names.append(orig_name)
                candidate_positions.append(pos_array)

    if len(candidate_names) < n_channels_to_add:
        raise ValueError(
            f"Not enough candidate channels in montage '{montage_source}'. "
            f"Need {n_channels_to_add} but only {len(candidate_names)} available."
        )

    # Use first epoch's positions as reference for current channels
    current_positions = positions_list[0] if len(positions_list) > 0 else np.zeros((0, 3))

    # Greedy selection: pick channels furthest from existing channels
    selected_indices = []
    selected_distances = []

    for cand_idx, cand_pos in enumerate(candidate_positions):
        # Compute minimum distance to any existing channel
        if len(current_positions) > 0:
            distances = np.linalg.norm(current_positions - cand_pos, axis=1)
            min_distance = np.min(distances)
        else:
            min_distance = np.inf

        selected_distances.append(min_distance)

    # Sort by distance (furthest first)
    sorted_indices = np.argsort(selected_distances)[::-1]

    # Select top N candidates
    selected_indices = sorted_indices[:n_channels_to_add]

    # Build new channel names and positions
    new_channel_names = [candidate_names[i] for i in selected_indices]
    new_channel_positions = np.array([candidate_positions[i] for i in selected_indices])

    # Combine with existing
    upsampled_names = list(channel_names) + new_channel_names
    upsampled_positions_ref = np.vstack([current_positions, new_channel_positions])

    # Upsample each epoch
    upsampled_epochs = []
    upsampled_positions = []

    n_times = epochs_list[0].shape[1] if len(epochs_list) > 0 else 0

    for epoch_idx, epoch in enumerate(epochs_list):
        current_epoch_channels = epoch.shape[0]
        current_epoch_pos = positions_list[epoch_idx]

        # Create new epoch with zeros for added channels
        new_epoch = np.zeros((current_epoch_channels + n_channels_to_add, n_times), dtype=epoch.dtype)
        new_epoch[:current_epoch_channels, :] = epoch

        # Combine positions
        new_positions = np.vstack([current_epoch_pos, new_channel_positions])

        upsampled_epochs.append(new_epoch)
        upsampled_positions.append(new_positions)

    return upsampled_epochs, upsampled_positions, upsampled_names
