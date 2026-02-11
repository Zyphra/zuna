"""
Channel upsampling utilities for EEG data.
"""
import numpy as np
import mne
from typing import List, Tuple, Optional


def zero_bad_channels(
    epochs_list: List[np.ndarray],
    channel_names: List[str],
    bad_channel_names: List[str]
) -> List[np.ndarray]:
    """
    Zero out specified bad channels in all epochs.

    This sets the data for specified channels to all zeros, effectively marking
    them as bad and forcing the model to interpolate them. The channels are NOT
    removed from the data - they remain in place but with zero values.

    Args:
        epochs_list: List of epoch arrays, each shape (n_channels, n_times)
        channel_names: List of channel names corresponding to the epochs
        bad_channel_names: List of channel names to zero out (e.g., ['Cz', 'Fz'])

    Returns:
        epochs_list: List of epochs with bad channels zeroed out

    Example:
        >>> epochs = zero_bad_channels(epochs, ['Fp1', 'Fp2', 'Cz'], ['Cz'])
        >>> # Now channel 'Cz' is all zeros in all epochs
    """
    if not bad_channel_names or len(epochs_list) == 0:
        return epochs_list

    # Normalize channel names (remove spaces, convert to lowercase)
    def normalize_name(name):
        return name.replace(' ', '').lower()

    # Create mapping of normalized names to original indices
    normalized_to_idx = {
        normalize_name(name): idx
        for idx, name in enumerate(channel_names)
    }

    # Find indices of bad channels
    bad_indices = []
    skipped_channels = []
    for bad_name in bad_channel_names:
        normalized = normalize_name(bad_name)
        if normalized in normalized_to_idx:
            bad_indices.append(normalized_to_idx[normalized])
        else:
            skipped_channels.append(bad_name)

    if not bad_indices:
        print(f"⚠️  Bad channels: No matching channels found. Skipped: {skipped_channels}")
        return epochs_list

    # Zero out bad channels in all epochs
    zeroed_count = 0
    for epoch in epochs_list:
        for idx in bad_indices:
            if idx < epoch.shape[0]:
                epoch[idx, :] = 0.0
                zeroed_count += 1

    print(f"✓ Zeroed out {len(bad_indices)} bad channels in {len(epochs_list)} epochs: {[channel_names[i] for i in bad_indices]}")
    if skipped_channels:
        print(f"  Skipped (not found): {skipped_channels}")

    return epochs_list


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


def add_specific_channels(
    epochs_list: List[np.ndarray],
    positions_list: List[np.ndarray],
    channel_names: List[str],
    target_channel_names: List[str],
    montage_source: str = 'standard_1005'
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Add specific channels by name with 3D coordinates from standard montage.

    New channels are added with all values set to 0, signaling the model to interpolate them.

    Parameters
    ----------
    epochs_list : list of np.ndarray
        List of epoch arrays, each (n_channels, n_times)
    positions_list : list of np.ndarray
        List of 3D channel position arrays, each (n_channels, 3)
    channel_names : list of str
        Current channel names
    target_channel_names : list of str
        List of channel names to add (e.g., ['Cz', 'Pz', 'Oz'])
    montage_source : str, optional
        MNE montage name to use for channel positions
        Default: 'standard_1005'

    Returns
    -------
    upsampled_epochs : list of np.ndarray
        Epoch arrays with added channels
    upsampled_positions : list of np.ndarray
        Position arrays with added channels
    upsampled_names : list of str
        Channel names after adding new channels

    Examples
    --------
    >>> epochs_up, positions_up, names_up = add_specific_channels(
    ...     epochs_list, positions_list, channel_names,
    ...     target_channel_names=['Cz', 'Pz', 'Oz']
    ... )
    """
    if len(epochs_list) == 0:
        return epochs_list, positions_list, channel_names

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

    # Process target channels
    new_channel_names = []
    new_channel_positions = []
    skipped_existing = []
    skipped_not_in_montage = []

    for target_name in target_channel_names:
        target_name_norm = target_name.lower()

        # Check if channel already exists
        if target_name_norm in current_names_normalized:
            skipped_existing.append(target_name)
            continue

        # Check if channel is in montage
        if target_name_norm not in montage_names_normalized:
            skipped_not_in_montage.append(target_name)
            continue

        # Get position from montage
        montage_orig_name = montage_names_normalized[target_name_norm]
        pos = montage_pos[montage_orig_name]
        pos_array = np.array([pos[0], pos[1], pos[2]])

        # Skip if position is all zeros
        if np.allclose(pos_array, [0.0, 0.0, 0.0]):
            skipped_not_in_montage.append(target_name)
            continue

        new_channel_names.append(montage_orig_name)
        new_channel_positions.append(pos_array)

    # Print summary
    if skipped_existing:
        print(f"  Skipped {len(skipped_existing)} channels (already exist): {', '.join(skipped_existing)}")

    if skipped_not_in_montage:
        print(f"  Skipped {len(skipped_not_in_montage)} channels (not in {montage_source}): {', '.join(skipped_not_in_montage)}")

    # If no new channels to add, return original
    if len(new_channel_names) == 0:
        print(f"  No new channels to add")
        return epochs_list, positions_list, channel_names

    print(f"  Adding {len(new_channel_names)} channels: {', '.join(new_channel_names)}")

    # Convert to numpy array
    new_channel_positions = np.array(new_channel_positions)

    # Use first epoch's positions as reference
    current_positions = positions_list[0] if len(positions_list) > 0 else np.zeros((0, 3))

    # Combine with existing
    upsampled_names = list(channel_names) + new_channel_names

    # Upsample each epoch
    upsampled_epochs = []
    upsampled_positions = []

    n_times = epochs_list[0].shape[1] if len(epochs_list) > 0 else 0

    for epoch_idx, epoch in enumerate(epochs_list):
        current_epoch_channels = epoch.shape[0]
        current_epoch_pos = positions_list[epoch_idx]

        # Create new epoch with zeros for added channels
        new_epoch = np.zeros((current_epoch_channels + len(new_channel_names), n_times), dtype=epoch.dtype)
        new_epoch[:current_epoch_channels, :] = epoch

        # Combine positions
        new_positions = np.vstack([current_epoch_pos, new_channel_positions])

        upsampled_epochs.append(new_epoch)
        upsampled_positions.append(new_positions)

    return upsampled_epochs, upsampled_positions, upsampled_names
