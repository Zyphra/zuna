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

    def normalize_epochs(self, epoch_data: np.ndarray, zero_mask: np.ndarray = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply final z-score normalization to cleaned epoch data.

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
        if zero_mask is None:
            # Normalize all data
            final_mean = float(epoch_data.mean())
            final_std = float(epoch_data.std())

            if final_std > 0:
                epoch_data_normalized = (epoch_data - final_mean) / final_std
            else:
                epoch_data_normalized = epoch_data.copy()
        else:
            # Normalize only non-zero data
            non_zero_data = epoch_data[~zero_mask]

            if len(non_zero_data) > 0:
                final_mean = float(non_zero_data.mean())
                final_std = float(non_zero_data.std())

                if final_std > 0:
                    epoch_data_normalized = epoch_data.copy()
                    epoch_data_normalized[~zero_mask] = (non_zero_data - final_mean) / final_std
                else:
                    epoch_data_normalized = epoch_data.copy()
            else:
                final_mean = 0.0
                final_std = 1.0
                epoch_data_normalized = epoch_data.copy()

        norm_params = {
            'type': 'final_zscore',
            'mean': final_mean,
            'std': final_std,
            'used_non_zero_only': zero_mask is not None,
            'n_samples_used': len(non_zero_data) if zero_mask is not None else epoch_data.size
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
            Complete normalization history for reconstruction
        """
        if not self.normalization_history:
            return {}

        # Extract the two main normalization steps
        params = {
            'normalization_chain': self.normalization_history
        }

        # Shortcut to main params for easier access
        if len(self.normalization_history) >= 2:
            params['global_mean'] = self.normalization_history[0]['mean']
            params['global_std'] = self.normalization_history[0]['std']
            params['final_mean'] = self.normalization_history[1]['mean']
            params['final_std'] = self.normalization_history[1]['std']

        return params

    @staticmethod
    def denormalize(data: np.ndarray, norm_params: Dict[str, Any]) -> np.ndarray:
        """
        Reverse normalization to reconstruct original scale.

        Parameters
        ----------
        data : np.ndarray
            Normalized data
        norm_params : dict
            Normalization parameters from get_reversibility_params()

        Returns
        -------
        data_original : np.ndarray
            Data in original scale
        """
        data_reconstructed = data.copy()

        # Reverse in opposite order: final → global
        if 'final_mean' in norm_params and 'final_std' in norm_params:
            data_reconstructed = data_reconstructed * norm_params['final_std'] + norm_params['final_mean']

        if 'global_mean' in norm_params and 'global_std' in norm_params:
            data_reconstructed = data_reconstructed * norm_params['global_std'] + norm_params['global_mean']

        return data_reconstructed
