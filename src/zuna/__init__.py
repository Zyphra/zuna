"""
Zuna: EEG Foundation Model and Preprocessing Pipeline
"""
from typing import Optional
import mne

from .preprocessing import ProcessingConfig, EEGProcessor, pt_to_raw as _pt_to_raw

__version__ = "0.1.0"


def raw_to_pt(raw: mne.io.Raw,
             output_path: str,
             config: Optional[ProcessingConfig] = None,
              **kwargs) -> dict:
    """
    Convert MNE Raw object to PT file with preprocessing.

    Simplified API for preprocessing EEG data. The raw object must have
    a montage set (with 3D channel coordinates).

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data with montage set
    output_path : str
        Path to save PT file
    config : ProcessingConfig, optional
        Configuration object. If None, uses defaults.
    **kwargs : keyword arguments
        Configuration overrides (e.g., drop_bad_channels=False)

    Returns
    -------
    metadata : dict
        Processing statistics and metadata

    Examples
    --------
    >>> import zuna
    >>> import mne
    >>>
    >>> # Load and set montage
    >>> raw = mne.io.read_raw_fif('data.fif', preload=True)
    >>> montage = mne.channels.make_standard_montage('standard_1005')
    >>> raw.set_montage(montage)
    >>>
    >>> # Process with defaults
    >>> metadata = zuna.raw_to_pt(raw, 'output.pt')
    >>>
    >>> # Or with custom settings
    >>> metadata = zuna.raw_to_pt(
    ...     raw, 'output.pt',
    ...     drop_bad_channels=True,
    ...     save_incomplete_batches=False
    ... )
    """
    # Create config with overrides
    if config is None:
        config = ProcessingConfig(**kwargs)
    elif kwargs:
        # Update config with keyword arguments
        config_dict = config.__dict__.copy()
        config_dict.update(kwargs)
        config = ProcessingConfig(**config_dict)

    # Create processor and run
    processor = EEGProcessor(config)
    metadata = processor.process_and_save(raw, output_path)

    return metadata


def pt_to_raw(pt_path: str) -> mne.io.Raw:
    """
    Convert PT file back to MNE Raw object.

    Reconstructs the original data scale using saved normalization parameters.

    Parameters
    ----------
    pt_path : str
        Path to PT file

    Returns
    -------
    raw : mne.io.Raw
        Reconstructed MNE Raw object

    Examples
    --------
    >>> import zuna
    >>>
    >>> # Reconstruct raw from PT file
    >>> raw = zuna.pt_to_raw('output.pt')
    >>>
    >>> # Now you can use standard MNE functions
    >>> raw.plot()
    >>> raw.compute_psd().plot()
    """
    return _pt_to_raw(pt_path)


# Expose key classes for advanced usage
from .preprocessing import ProcessingConfig, EEGProcessor

__all__ = [
    'raw_to_pt',
    'pt_to_raw',
    'ProcessingConfig',
    'EEGProcessor',
]
