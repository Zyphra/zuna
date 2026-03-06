"""
Normalization with reversibility support for pt_to_raw reconstruction.
"""
import numpy as np
from typing import Tuple, Dict, Any
import mne


class Normalizer:
    """Handles z-score normalization with reversibility tracking."""

    def __init__(self, save_params: bool = True):
        """
        Parameters
        ----------
        save_params : bool
            Whether to save normalization parameters for later reconstruction
        """
        self.save_params = save_params
        self.normalization_history = []

    def normalize_raw(self, raw: mne.io.Raw) -> Tuple[mne.io.Raw, Dict[str, Any]]:
        """
        Apply global z-score normalization to raw data.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data

        Returns
        -------
        raw_normalized : mne.io.Raw
            Normalized raw data (modified in place)
        norm_params : dict
            Dictionary with normalization parameters for reversibility
        """
        # Get good channels only for computing stats
        good_picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, stim=False, exclude='bads')

        if len(good_picks) == 0:
            raise ValueError("No good EEG channels remaining for normalization")

        # Compute global mean and std from good channels only
        good_data = raw.get_data(picks=good_picks)
        global_mean = float(good_data.mean())
        global_std = float(good_data.std())

        if global_std == 0:
            raise ValueError("Global std is zero; data appears constant")

        # Apply normalization to ALL channels (good + bad)
        all_picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, stim=False, exclude=[])
        all_data = raw.get_data(picks=all_picks)
        data_z = (all_data - global_mean) / global_std
        raw._data[all_picks] = data_z

        norm_params = {
            'type': 'global_zscore',
            'mean': global_mean,
            'std': global_std,
            'n_channels_used': len(good_picks),
            'channel_names': [raw.ch_names[i] for i in all_picks]
        }

        if self.save_params:
            self.normalization_history.append(norm_params)

        return raw, norm_params

    def normalize_epoch_array(self, epoch_data: np.ndarray,
                              bad_channels: set = None,
                              channel_names: list = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply per-epoch, per-channel z-score normalization.

        Each channel in each epoch is normalized independently.
        Params stored as (n_epochs, n_channels) matrices for reversibility.

        Parameters
        ----------
        epoch_data : np.ndarray
            Epoch data (n_epochs, n_channels, n_times)
        bad_channels : set, optional
            Channel names to exclude from stats computation (not used for
            per-channel norm, but kept for API compatibility)
        channel_names : list, optional
            Channel names

        Returns
        -------
        epoch_data_normalized : np.ndarray
            Normalized epoch data
        norm_params : dict
            Normalization parameters for reversibility
        """
        n_epochs, n_channels, n_times = epoch_data.shape

        # Compute per-epoch, per-channel mean and std
        # means/stds shape: (n_epochs, n_channels)
        means = epoch_data.mean(axis=2)  # mean over time
        stds = epoch_data.std(axis=2)    # std over time
        stds[stds == 0] = 1.0  # avoid division by zero

        # Normalize: broadcast (n_epochs, n_channels, 1)
        epoch_data_normalized = (epoch_data - means[:, :, np.newaxis]) / stds[:, :, np.newaxis]

        norm_params = {
            'type': 'per_epoch_channel_zscore',
            'means': means.tolist(),  # list of lists: [n_epochs][n_channels]
            'stds': stds.tolist(),
        }

        if self.save_params:
            self.normalization_history.append(norm_params)

        return epoch_data_normalized, norm_params

    def normalize_epochs(self, epoch_data: np.ndarray, zero_mask: np.ndarray = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply per-epoch, per-channel final z-score normalization.

        Each channel in each epoch is normalized independently.
        Zeroed samples are excluded from stats per channel per epoch.

        Parameters
        ----------
        epoch_data : np.ndarray
            Epoch data array (n_epochs, n_channels, n_times)
        zero_mask : np.ndarray, optional
            Boolean mask of zeroed samples to exclude from normalization

        Returns
        -------
        epoch_data_normalized : np.ndarray
            Normalized epoch data
        norm_params : dict
            Normalization parameters
        """
        n_epochs, n_channels, n_times = epoch_data.shape
        means = np.zeros((n_epochs, n_channels), dtype=np.float64)
        stds = np.zeros((n_epochs, n_channels), dtype=np.float64)
        epoch_data_normalized = epoch_data.copy()

        if zero_mask is None:
            # Fast path: no mask
            means = epoch_data.mean(axis=2)
            stds = epoch_data.std(axis=2)
            stds[stds == 0] = 1.0
            epoch_data_normalized = (epoch_data - means[:, :, np.newaxis]) / stds[:, :, np.newaxis]
        else:
            for i in range(n_epochs):
                for ch in range(n_channels):
                    mask_ch = ~zero_mask[i, ch, :]
                    if mask_ch.any():
                        vals = epoch_data[i, ch, mask_ch]
                        means[i, ch] = vals.mean()
                        stds[i, ch] = vals.std()
                        if stds[i, ch] > 0:
                            epoch_data_normalized[i, ch, mask_ch] = (vals - means[i, ch]) / stds[i, ch]
                    else:
                        stds[i, ch] = 1.0

        norm_params = {
            'type': 'per_epoch_channel_final_zscore',
            'means': means.tolist(),
            'stds': stds.tolist(),
            'used_non_zero_only': zero_mask is not None,
        }

        if self.save_params:
            self.normalization_history.append(norm_params)

        return epoch_data_normalized, norm_params

    def get_reversibility_params(self) -> Dict[str, Any]:
        """
        Get all normalization parameters needed for reversibility.

        Returns
        -------
        params : dict
            Complete normalization history for reconstruction.
            For per-epoch normalization, contains 'global_means', 'global_stds',
            'final_means', 'final_stds' as lists.
            For global normalization, contains scalar 'global_mean', 'global_std',
            'final_mean', 'final_std'.
        """
        if not self.normalization_history:
            return {}

        params = {
            'normalization_chain': self.normalization_history
        }

        # Extract shortcuts based on normalization type
        if len(self.normalization_history) >= 2:
            step0 = self.normalization_history[0]
            step1 = self.normalization_history[1]

            if 'means' in step0:
                # Per-epoch normalization
                params['global_means'] = step0['means']
                params['global_stds'] = step0['stds']
                params['final_means'] = step1['means']
                params['final_stds'] = step1['stds']
            else:
                # Global normalization (original raw path)
                params['global_mean'] = step0['mean']
                params['global_std'] = step0['std']
                params['final_mean'] = step1['mean']
                params['final_std'] = step1['std']

        return params

    @staticmethod
    def denormalize(data: np.ndarray, norm_params: Dict[str, Any]) -> np.ndarray:
        """
        Reverse normalization to reconstruct original scale.

        Handles both global (scalar) and per-epoch (array) normalization params.

        Parameters
        ----------
        data : np.ndarray
            Normalized data. For per-epoch: (n_epochs, n_channels, n_times)
        norm_params : dict
            Normalization parameters from get_reversibility_params()

        Returns
        -------
        data_original : np.ndarray
            Data in original scale
        """
        data_reconstructed = data.copy()

        # Check if per-epoch-channel, per-epoch, or global
        if 'final_means' in norm_params and 'final_stds' in norm_params:
            final_means = np.array(norm_params['final_means'])
            final_stds = np.array(norm_params['final_stds'])

            if final_means.ndim == 2:
                # Per-epoch, per-channel: shape (n_epochs, n_channels_norm)
                # Data may have more channels (upsampled), denorm real channels
                # and scale upsampled channels by average std
                n_ep = min(final_means.shape[0], data_reconstructed.shape[0])
                n_ch_norm = final_means.shape[1]
                n_ch_data = data_reconstructed.shape[1]
                n_ch = min(n_ch_norm, n_ch_data)
                # Denorm real channels
                data_reconstructed[:n_ep, :n_ch, :] = (
                    data_reconstructed[:n_ep, :n_ch, :] * final_stds[:n_ep, :n_ch, np.newaxis]
                    + final_means[:n_ep, :n_ch, np.newaxis]
                )
                # Scale upsampled channels by average std (keep mean=0)
                if n_ch_data > n_ch_norm:
                    avg_std = final_stds[:n_ep, :n_ch].mean(axis=1, keepdims=True)  # (n_ep, 1)
                    data_reconstructed[:n_ep, n_ch_norm:, :] = (
                        data_reconstructed[:n_ep, n_ch_norm:, :] * avg_std[:, :, np.newaxis]
                    )
            else:
                # Per-epoch only: shape (n_epochs,)
                for i in range(min(len(final_means), data_reconstructed.shape[0])):
                    data_reconstructed[i] = data_reconstructed[i] * final_stds[i] + final_means[i]

            if 'global_means' in norm_params and 'global_stds' in norm_params:
                global_means = np.array(norm_params['global_means'])
                global_stds = np.array(norm_params['global_stds'])

                if global_means.ndim == 2:
                    n_ep = min(global_means.shape[0], data_reconstructed.shape[0])
                    n_ch_norm = global_means.shape[1]
                    n_ch_data = data_reconstructed.shape[1]
                    n_ch = min(n_ch_norm, n_ch_data)
                    # Denorm real channels
                    data_reconstructed[:n_ep, :n_ch, :] = (
                        data_reconstructed[:n_ep, :n_ch, :] * global_stds[:n_ep, :n_ch, np.newaxis]
                        + global_means[:n_ep, :n_ch, np.newaxis]
                    )
                    # Scale upsampled channels by average std
                    if n_ch_data > n_ch_norm:
                        avg_std = global_stds[:n_ep, :n_ch].mean(axis=1, keepdims=True)
                        data_reconstructed[:n_ep, n_ch_norm:, :] = (
                            data_reconstructed[:n_ep, n_ch_norm:, :] * avg_std[:, :, np.newaxis]
                        )
                else:
                    for i in range(min(len(global_means), data_reconstructed.shape[0])):
                        data_reconstructed[i] = data_reconstructed[i] * global_stds[i] + global_means[i]
        else:
            # Global denormalization (original raw path)
            if 'final_mean' in norm_params and 'final_std' in norm_params:
                data_reconstructed = data_reconstructed * norm_params['final_std'] + norm_params['final_mean']

            if 'global_mean' in norm_params and 'global_std' in norm_params:
                data_reconstructed = data_reconstructed * norm_params['global_std'] + norm_params['global_mean']

        return data_reconstructed

