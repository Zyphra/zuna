"""
Bad channel and bad epoch detection.
"""
import numpy as np
import mne
from typing import Set, Dict, Any


class ArtifactRemover:
    """Detects and handles bad channels and bad epochs."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : ProcessingConfig
            Configuration object
        """
        self.config = config
        self.stats = {
            'channels_dropped_flat': 0,
            'channels_dropped_clipped': 0,
            'channels_dropped_noisy': 0,
            'epochs_dropped_outlier': 0,
            'epochs_dropped_amplitude': 0,
            'epochs_dropped_bad_channels': 0,
        }

    def detect_bad_channels(self, raw: mne.io.Raw) -> Set[str]:
        """
        Detect bad channels (flat, clipped, or noisy).

        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data (should already be normalized)

        Returns
        -------
        bad_channels : set
            Set of bad channel names
        """
        if not self.config.drop_bad_channels:
            return set()

        bads = set()
        picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, stim=False, exclude=[])
        data = raw.get_data(picks=picks)

        # Compute robust statistics
        all_stds = np.array([data[i].std() for i in range(len(picks))])
        median_std = np.median(all_stds)
        mad_std = np.median(np.abs(all_stds - median_std))

        flat_threshold = median_std - 3.0 * mad_std if mad_std > 0 else median_std * 0.01

        flat_channels = []
        clipped_channels = []
        noisy_channels = []

        # Detect flat and clipped channels
        for i in range(len(picks)):
            x = data[i]
            ch_sd = x.std()
            ch_name = raw.ch_names[picks[i]]

            # Check if channel is essentially flat
            if ch_sd < max(flat_threshold, median_std * 1e-6):
                bads.add(ch_name)
                flat_channels.append(ch_name)
                continue

            # Check for clipping (samples stuck at min/max)
            eps = 1e-3 * ch_sd
            near_max = np.isclose(x, x.max(), atol=eps)
            near_min = np.isclose(x, x.min(), atol=eps)
            frac = (near_max | near_min).mean()
            if frac > 0.005:  # >0.5% samples hugging min/max
                bads.add(ch_name)
                clipped_channels.append(ch_name)

        # Detect noisy channels (after removing flat/clipped)
        good_picks = [i for i in picks if raw.ch_names[i] not in bads]
        if len(good_picks) > 0:
            good_data = raw.get_data(picks=good_picks)
            channel_stds = np.std(good_data, axis=1)
            mean_std = np.mean(channel_stds)
            noisy_threshold = self.config.noisy_channel_threshold * mean_std

            for i, ch_std in enumerate(channel_stds):
                if ch_std > noisy_threshold:
                    ch_name = raw.ch_names[good_picks[i]]
                    noisy_channels.append(ch_name)
                    bads.add(ch_name)

        # Update statistics
        self.stats['channels_dropped_flat'] = len(flat_channels)
        self.stats['channels_dropped_clipped'] = len(clipped_channels)
        self.stats['channels_dropped_noisy'] = len(noisy_channels)

        return bads

    def zero_out_artifacts(self, epoch_data: np.ndarray, bad_channels: Set[str],
                          channel_names: list) -> tuple[np.ndarray, np.ndarray]:
        """
        Zero out bad channels and artifact samples in epochs.

        Parameters
        ----------
        epoch_data : np.ndarray
            Epoch data (n_epochs, n_channels, n_times)
        bad_channels : set
            Set of bad channel names
        channel_names : list
            List of channel names corresponding to epoch_data channels

        Returns
        -------
        epoch_data_cleaned : np.ndarray
            Cleaned epoch data with artifacts zeroed
        zero_mask : np.ndarray
            Boolean mask of zeroed samples
        """
        if not self.config.zero_out_artifacts:
            return epoch_data, np.zeros_like(epoch_data, dtype=bool)

        epoch_data_cleaned = epoch_data.copy()
        zero_mask = np.zeros_like(epoch_data, dtype=bool)

        # Zero out bad channels
        for bad_ch in bad_channels:
            if bad_ch in channel_names:
                ch_idx = channel_names.index(bad_ch)
                epoch_data_cleaned[:, ch_idx, :] = 0.0
                zero_mask[:, ch_idx, :] = True

        # Detect outlier channel/epoch combinations
        std = np.std(epoch_data, axis=2)
        overall_mean_std = np.mean(std)

        high_threshold = overall_mean_std * self.config.outlier_std_multiplier
        low_threshold = overall_mean_std * 0.1

        high_outliers = std > high_threshold
        low_outliers = std < low_threshold
        outliers = high_outliers | low_outliers

        # Zero out outliers
        mask_full = np.broadcast_to(outliers[..., None], epoch_data_cleaned.shape)
        epoch_data_cleaned[mask_full] = 0.0
        zero_mask[mask_full] = True

        # Peak-to-peak amplitude detection
        peak_to_peak = np.max(epoch_data, axis=2) - np.min(epoch_data, axis=2)
        amplitude_outliers = peak_to_peak > self.config.bad_epoch_amplitude_threshold
        mask_amplitude = np.broadcast_to(amplitude_outliers[..., None], epoch_data_cleaned.shape)
        epoch_data_cleaned[mask_amplitude] = 0.0
        zero_mask[mask_amplitude] = True

        # Track statistics
        self.stats['epochs_dropped_outlier'] = int(np.sum(outliers))
        self.stats['epochs_dropped_amplitude'] = int(np.sum(amplitude_outliers & ~outliers))

        return epoch_data_cleaned, zero_mask

    def remove_bad_epochs(self, epoch_data: np.ndarray, zero_mask: np.ndarray) -> np.ndarray:
        """
        Remove entire epochs if >50% of channels are zeroed.

        Parameters
        ----------
        epoch_data : np.ndarray
            Cleaned epoch data
        zero_mask : np.ndarray
            Mask of zeroed samples

        Returns
        -------
        epoch_data_final : np.ndarray
            Final epoch data with bad epochs zeroed
        """
        if not self.config.drop_bad_epochs:
            return epoch_data

        # Count non-zero channels per epoch
        channels_per_epoch = zero_mask.shape[1]
        non_zero_channels_per_epoch = np.sum(~np.all(zero_mask, axis=2), axis=1)
        fraction_good = non_zero_channels_per_epoch / channels_per_epoch

        bad_epochs_mask = fraction_good < 0.5
        n_bad_epochs = np.sum(bad_epochs_mask)

        epoch_data_final = epoch_data.copy()
        epoch_data_final[bad_epochs_mask] = 0.0

        self.stats['epochs_dropped_bad_channels'] = int(n_bad_epochs)

        return epoch_data_final

    def get_stats(self) -> Dict[str, Any]:
        """Get artifact removal statistics."""
        return self.stats.copy()
