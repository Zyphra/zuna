"""
Configuration for EEG preprocessing pipeline.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProcessingConfig:
    """Configuration for EEG preprocessing.

    Parameters
    ----------
    drop_bad_channels : bool
        Whether to detect and drop bad channels (flat, clipped, noisy)
    drop_bad_epochs : bool
        Whether to detect and drop bad epochs
    apply_notch_filter : bool
        Whether to apply automatic notch filtering (detects line noise peaks)
    apply_highpass_filter : bool
        Whether to apply highpass filter
    zero_out_artifacts : bool
        Whether to zero out artifact samples (outliers, high amplitude)

    target_sfreq : float
        Target sampling rate in Hz
    epoch_duration : float
        Duration of each epoch in seconds
    epochs_per_file : int
        Number of epochs to save per PT file

    save_incomplete_batches : bool
        If True, save remaining epochs even if < epochs_per_file
        If False, discard incomplete batches
    min_epochs_to_save : int
        Minimum number of epochs required to save (only used if save_incomplete_batches=True)

    hpf_freq : float
        Highpass filter frequency in Hz
    bad_epoch_amplitude_threshold : float
        Peak-to-peak amplitude threshold for bad epoch detection (in z-score units)
    outlier_std_multiplier : float
        Multiplier for outlier detection (epochs with std > mean_std * multiplier are flagged)

    save_normalization_params : bool
        Whether to save normalization parameters for reversibility (pt_to_raw)
    """

    # Processing toggles
    drop_bad_channels: bool = True
    drop_bad_epochs: bool = True
    apply_notch_filter: bool = True
    apply_highpass_filter: bool = True
    zero_out_artifacts: bool = True

    # Basic parameters
    target_sfreq: float = 256.0
    epoch_duration: float = 5.0
    epochs_per_file: int = 64

    # Incomplete batch handling
    save_incomplete_batches: bool = True
    min_epochs_to_save: int = 1

    # Filtering parameters
    hpf_freq: float = 0.5

    # Artifact detection parameters
    bad_epoch_amplitude_threshold: float = 20.0
    outlier_std_multiplier: float = 3.0
    noisy_channel_threshold: float = 3.0

    # Reversibility
    save_normalization_params: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.target_sfreq <= 0:
            raise ValueError("target_sfreq must be positive")
        if self.epoch_duration <= 0:
            raise ValueError("epoch_duration must be positive")
        if self.epochs_per_file <= 0:
            raise ValueError("epochs_per_file must be positive")
        if self.min_epochs_to_save < 1:
            raise ValueError("min_epochs_to_save must be at least 1")
        if self.min_epochs_to_save > self.epochs_per_file:
            raise ValueError("min_epochs_to_save cannot exceed epochs_per_file")
