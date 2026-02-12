"""
Main EEG preprocessing processor.
"""
import mne
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import warnings

from .config import ProcessingConfig
from .normalizer import Normalizer
from .artifact_removal import ArtifactRemover
from .filtering import Filter
from .io import save_pt, epochs_to_list


warnings.filterwarnings("ignore")
mne.set_log_level('ERROR')


class EEGProcessor:
    """
    Main preprocessing processor for EEG data.

    Simplified interface that expects raw data with montage already set.
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Parameters
        ----------
        config : ProcessingConfig, optional
            Configuration object. If None, uses defaults.
        """
        self.config = config if config is not None else ProcessingConfig()
        self.normalizer = Normalizer(save_params=self.config.save_normalization_params)
        self.artifact_remover = ArtifactRemover(self.config)
        self.filter = Filter(self.config)
        self.stats = {}

    def process(self, raw: mne.io.Raw) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
        """
        Process raw EEG data through full pipeline.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data with montage already set

        Returns
        -------
        epochs_list : list of np.ndarray
            List of processed epochs
        positions_list : list of np.ndarray
            List of 3D channel positions for each epoch
        metadata : dict
            Processing metadata including statistics and reversibility params
        """
        # Validate input
        if raw.get_montage() is None:
            raise ValueError("Raw data must have a montage set. Use raw.set_montage() first.")

        raw = raw.copy()  # Don't modify original

        # Save original filename BEFORE any modifications (RawArray creation loses filenames)
        import os
        if raw.filenames and raw.filenames[0] is not None:
            original_filename = os.path.basename(str(raw.filenames[0]))
        else:
            original_filename = "preprocessed_raw.fif"

        # Track bad channels from original raw.info['bads'] (if enabled)
        channels_marked_bad_in_raw = set(raw.info['bads']) if self.config.zero_bad_channels_from_raw else set()

        # Get original info for metadata
        orig_sfreq = float(raw.info['sfreq'])
        orig_n_channels = len(raw.ch_names)
        orig_duration = float(raw.n_times / raw.info['sfreq'])

        # Extract 3D channel positions from montage
        montage = raw.get_montage()
        ch_pos = montage.get_positions()['ch_pos']

        # Drop channels without 3D coordinates
        channels_with_coords = []
        channel_positions_dict = {}
        for ch_name in raw.ch_names:
            if ch_name in ch_pos:
                pos = ch_pos[ch_name]
                if not np.allclose(pos, [0.0, 0.0, 0.0]):
                    channels_with_coords.append(ch_name)
                    channel_positions_dict[ch_name] = np.array(pos)

        channels_dropped_no_coords = [ch for ch in raw.ch_names if ch not in channels_with_coords]

        if len(channels_with_coords) == 0:
            raise ValueError("No channels with valid 3D coordinates found")

        # Keep only channels with coordinates
        raw.pick_channels(channels_with_coords)

        # Create position array (ordered by current channel order)
        channel_positions = np.array([channel_positions_dict[ch] for ch in raw.ch_names])

        # Resample
        raw = self.filter.resample(raw)

        # Apply filtering BEFORE normalization
        # This ensures filtered data is used for both preprocessing and comparison
        raw = self.filter.apply_highpass(raw)
        raw = self.filter.apply_reference(raw)
        raw, notch_freqs = self.filter.apply_notch(raw)

        # Save preprocessed FIF AFTER filtering but BEFORE normalization
        # Zero out bad channels in raw data if specified (before saving preprocessed FIF)
        if self.config.bad_channels is not None:
            # Normalize channel names for matching
            def normalize_name(name):
                return name.replace(' ', '').lower()

            # Find indices of bad channels
            ch_names_normalized = {normalize_name(name): idx for idx, name in enumerate(raw.ch_names)}
            bad_indices = []
            for bad_name in self.config.bad_channels:
                normalized = normalize_name(bad_name)
                if normalized in ch_names_normalized:
                    bad_indices.append(ch_names_normalized[normalized])

            # Zero out bad channels in raw data
            if bad_indices:
                raw_data = raw.get_data()
                for idx in bad_indices:
                    raw_data[idx, :] = 0.0
                raw = mne.io.RawArray(raw_data, raw.info, verbose=False)
                print(f"✓ Zeroed out {len(bad_indices)} bad channels in raw data: {[raw.ch_names[i] for i in bad_indices]}")

        # This ensures the preprocessed FIF has the same filtering as the model input,
        # but is in the original scale (not normalized) for comparison
        if self.config.save_preprocessed_fif and self.config.preprocessed_fif_dir:
            from pathlib import Path
            preprocessed_dir = Path(self.config.preprocessed_fif_dir)
            preprocessed_dir.mkdir(parents=True, exist_ok=True)

            # Use the saved original filename (captured at line 67 before RawArray creation)
            preprocessed_path = preprocessed_dir / original_filename

            # Save filtered raw data BEFORE normalization (original scale, filtered)
            raw.save(str(preprocessed_path), overwrite=True, verbose=False)

        # Initial normalization (global z-score)
        raw, norm_params_1 = self.normalizer.normalize_raw(raw)

        # Bad channel detection (on normalized data for better detection)
        bad_channels = self.artifact_remover.detect_bad_channels(raw)
        raw.info['bads'] = sorted(list(bad_channels))

        # Create epochs
        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=self.config.epoch_duration,
            preload=True,
            verbose=False
        )
        epochs.apply_baseline((None, None))
        epoch_data = epochs.get_data()

        # Artifact removal
        epoch_data_cleaned, zero_mask = self.artifact_remover.zero_out_artifacts(
            epoch_data, bad_channels, raw.ch_names
        )

        # Remove bad epochs
        epoch_data_cleaned = self.artifact_remover.remove_bad_epochs(epoch_data_cleaned, zero_mask)

        # Zero out channels from raw.info['bads'] BEFORE final normalization
        # This ensures normalization stats are computed correctly
        for ch_idx, ch_name in enumerate(raw.ch_names):
            if ch_name in channels_marked_bad_in_raw:
                epoch_data_cleaned[:, ch_idx, :] = 0.0
                zero_mask[:, ch_idx, :] = True

        # Final normalization (after all channels are zeroed)
        epoch_data_cleaned, norm_params_2 = self.normalizer.normalize_epochs(
            epoch_data_cleaned, zero_mask
        )

        # Convert to list format (keep all channels for consistency)
        # Note: channels are already zeroed above, so we don't pass zero_channels here
        epochs_list, positions_list = epochs_to_list(
            epoch_data_cleaned,
            channel_positions,
            remove_all_zero=False,  # Keep all channels for consistent structure
            zero_channels=None,  # Already zeroed above
            channel_names=raw.ch_names
        )

        # Zero out bad channels if specified
        channel_names_final = list(raw.ch_names)  # Convert to list for potential modification
        if self.config.bad_channels is not None and len(epochs_list) > 0:
            from .interpolation import zero_bad_channels
            epochs_list = zero_bad_channels(
                epochs_list,
                channel_names_final,
                bad_channel_names=self.config.bad_channels
            )

        # Apply upsampling if configured
        if self.config.target_channel_count is not None and len(epochs_list) > 0:
            current_n_channels = len(channel_names_final)

            # Check if target_channel_count is an int or a list
            if isinstance(self.config.target_channel_count, int):
                # Mode 1: Upsample to target number of channels (greedy selection)
                from .interpolation import upsample_channels
                if current_n_channels < self.config.target_channel_count:
                    epochs_list, positions_list, channel_names_final = upsample_channels(
                        epochs_list,
                        positions_list,
                        channel_names_final,
                        target_n_channels=self.config.target_channel_count
                    )
            elif isinstance(self.config.target_channel_count, list):
                # Mode 2: Add specific channels by name
                from .interpolation import add_specific_channels
                epochs_list, positions_list, channel_names_final = add_specific_channels(
                    epochs_list,
                    positions_list,
                    channel_names_final,
                    target_channel_names=self.config.target_channel_count
                )

        # Check if we have enough epochs to save
        if not self.config.save_incomplete_batches:
            if len(epochs_list) < self.config.epochs_per_file:
                epochs_list = []
                positions_list = []
        elif len(epochs_list) < self.config.min_epochs_to_save:
            epochs_list = []
            positions_list = []

        # Build metadata
        metadata = {
            'original_sfreq': orig_sfreq,
            'resampled_sfreq': self.config.target_sfreq,
            'original_n_channels': orig_n_channels,
            'final_n_channels': len(channel_names_final),
            'original_duration_sec': orig_duration,
            'channel_names': channel_names_final,
            'channels_dropped_no_coords': channels_dropped_no_coords,
            'bad_channels': sorted(list(bad_channels)),
            'channels_zeroed_from_raw': sorted(list(channels_marked_bad_in_raw)),
            'notch_frequencies': notch_freqs,
            'n_epochs_original': len(epochs),
            'n_epochs_saved': len(epochs_list),
            'artifact_stats': self.artifact_remover.get_stats(),
        }

        # Add reversibility params
        if self.config.save_normalization_params:
            metadata['reversibility'] = self.normalizer.get_reversibility_params()

        self.stats = metadata.copy()

        return epochs_list, positions_list, metadata

    def process_and_save(self, raw: mne.io.Raw, output_path: str) -> Dict[str, Any]:
        """
        Process raw data and save to PT file.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data with montage set
        output_path : str
            Path to save PT file

        Returns
        -------
        metadata : dict
            Processing metadata
        """
        epochs_list, positions_list, metadata = self.process(raw)

        if len(epochs_list) == 0:
            raise ValueError(f"No epochs to save (had {metadata['n_epochs_original']} epochs, "
                           f"but all were removed or batch size requirements not met)")

        save_pt(
            epochs_list,
            positions_list,
            metadata['channel_names'],
            output_path,
            metadata=metadata,
            reversibility_params=metadata.get('reversibility')
        )

        return metadata

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()
