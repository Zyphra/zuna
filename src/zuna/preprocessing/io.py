"""
PT file I/O with reversibility support.
"""
import torch
import numpy as np
import mne
from typing import Dict, List, Any, Optional
from pathlib import Path


def save_pt(epochs_list: List[np.ndarray],
           positions_list: List[np.ndarray],
           channel_names: List[str],
           output_path: str,
           metadata: Optional[Dict[str, Any]] = None,
           reversibility_params: Optional[Dict[str, Any]] = None) -> None:
    """
    Save processed epochs to PT file with reversibility metadata.

    Parameters
    ----------
    epochs_list : list of np.ndarray
        List of epoch arrays, each (n_channels, n_times)
    positions_list : list of np.ndarray
        List of 3D channel position arrays, each (n_channels, 3)
    channel_names : list of str
        Channel names
    output_path : str
        Path to save PT file
    metadata : dict, optional
        Additional metadata to store
    reversibility_params : dict, optional
        Normalization parameters for pt_to_raw reconstruction
    """
    # Convert to tensors
    data_tensors = [torch.tensor(epoch, dtype=torch.float32) for epoch in epochs_list]
    position_tensors = [torch.tensor(pos, dtype=torch.float32) for pos in positions_list]

    # Build metadata
    meta = {
        'n_epochs': len(epochs_list),
        'channel_names': channel_names,
        'avg_channels_per_epoch': float(np.mean([ep.shape[0] for ep in epochs_list])),
        'samples_per_epoch': epochs_list[0].shape[1] if len(epochs_list) > 0 else 0,
    }

    if metadata is not None:
        meta.update(metadata)

    # Add reversibility info
    if reversibility_params is not None:
        meta['reversibility'] = reversibility_params

    # Create data dictionary
    data_dict = {
        'data': data_tensors,
        'channel_positions': position_tensors,
        'metadata': meta
    }

    # Save
    torch.save(data_dict, output_path)


def load_pt(pt_path: str) -> Dict[str, Any]:
    """
    Load PT file.

    Parameters
    ----------
    pt_path : str
        Path to PT file

    Returns
    -------
    data : dict
        Dictionary with 'data', 'channel_positions', 'metadata' keys
    """
    return torch.load(pt_path, weights_only=False)


def pt_to_raw(pt_path: str) -> mne.io.Raw:
    """
    Convert PT file back to MNE Raw object.

    This reconstructs the original scale using saved normalization parameters.

    Parameters
    ----------
    pt_path : str
        Path to PT file

    Returns
    -------
    raw : mne.io.Raw
        Reconstructed MNE Raw object
    """
    from .normalizer import Normalizer

    pt_data = load_pt(pt_path)
    metadata = pt_data['metadata']

    # Extract data - handle both tensors and numpy arrays, filter out None values
    epochs_list = [tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
                   for tensor in pt_data['data'] if tensor is not None]
    positions_list = [tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
                      for tensor in pt_data['channel_positions'] if tensor is not None]

    # Get channel info
    if 'channel_names' in metadata:
        channel_names = metadata['channel_names']
    else:
        # Fallback: use first epoch's channel count
        n_channels = epochs_list[0].shape[0]
        channel_names = [f'Ch{i+1}' for i in range(n_channels)]

    # Get sampling rate
    sfreq = metadata.get('sampling_rate', metadata.get('resampled_sampling_rate', 256.0))

    # Concatenate epochs into continuous data
    # Each epoch might have different number of channels (some were zeroed)
    # We need to handle this carefully

    # Find maximum number of channels across all epochs
    max_channels = max(epoch.shape[0] for epoch in epochs_list)

    # Pad epochs to have same number of channels
    epochs_padded = []
    for epoch in epochs_list:
        if epoch.shape[0] < max_channels:
            # Pad with zeros
            padded = np.zeros((max_channels, epoch.shape[1]))
            padded[:epoch.shape[0], :] = epoch
            epochs_padded.append(padded)
        else:
            epochs_padded.append(epoch)

    # Concatenate along time axis
    continuous_data = np.concatenate(epochs_padded, axis=1)

    # Denormalize if reversibility params available
    if 'reversibility' in metadata:
        rev_params = metadata['reversibility']
        continuous_data = Normalizer.denormalize(continuous_data, rev_params)

    # Create MNE info
    # Use channel names from metadata, pad if needed
    if len(channel_names) < max_channels:
        channel_names = channel_names + [f'Ch{i+1}' for i in range(len(channel_names), max_channels)]
    elif len(channel_names) > max_channels:
        channel_names = channel_names[:max_channels]

    info = mne.create_info(
        ch_names=channel_names[:max_channels],
        sfreq=sfreq,
        ch_types='eeg'
    )

    # Create Raw object
    raw = mne.io.RawArray(continuous_data, info, verbose=False)

    # Set montage if positions available
    try:
        # Use positions from first epoch (they should all be similar)
        if len(positions_list) > 0:
            positions_3d = positions_list[0]
            if positions_3d.shape[0] == len(channel_names):
                ch_pos = {ch: pos for ch, pos in zip(channel_names, positions_3d)}
                montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
                raw.set_montage(montage, verbose=False)
    except Exception:
        pass  # Continue without montage if fails

    return raw


def epochs_to_list(epoch_data: np.ndarray,
                   channel_positions: np.ndarray,
                   remove_all_zero: bool = True,
                   zero_channels: set = None,
                   channel_names: List[str] = None) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Convert epoch array to list format, optionally removing all-zero entries.

    Parameters
    ----------
    epoch_data : np.ndarray
        Epoch data (n_epochs, n_channels, n_times)
    channel_positions : np.ndarray
        Channel positions (n_channels, 3)
    remove_all_zero : bool
        Whether to remove all-zero epochs/channels
    zero_channels : set, optional
        Set of channel names to zero out (for bad channels from raw.info['bads'])
    channel_names : list of str, optional
        List of channel names (must be provided if zero_channels is given)

    Returns
    -------
    epochs_list : list of np.ndarray
        List of epochs with variable channel counts
    positions_list : list of np.ndarray
        List of corresponding position arrays
    """
    # Zero out specified channels (bad channels from raw)
    if zero_channels and channel_names:
        for ch_idx, ch_name in enumerate(channel_names):
            if ch_name in zero_channels:
                epoch_data[:, ch_idx, :] = 0.0

    epochs_list = []
    positions_list = []

    for epoch_idx in range(epoch_data.shape[0]):
        epoch = epoch_data[epoch_idx]  # (channels, samples)

        if remove_all_zero:
            # Remove all-zero channels
            non_zero_channels = ~np.all(epoch == 0, axis=1)

            if np.sum(non_zero_channels) == 0:
                continue  # Skip this epoch entirely

            clean_epoch = epoch[non_zero_channels]
            clean_positions = channel_positions[non_zero_channels]
        else:
            clean_epoch = epoch
            clean_positions = channel_positions

        epochs_list.append(clean_epoch)
        positions_list.append(clean_positions)

    return epochs_list, positions_list
