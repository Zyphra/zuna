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


def pt_directory_to_fif(input_dir: str, output_dir: str) -> dict:
    """
    Convert all PT files in a directory to FIF format.

    This function intelligently reconstructs original FIF files by:
    1. Reading metadata from each PT file to find the original source filename
    2. Grouping PT files that came from the same source file
    3. Stitching them back together into a single continuous FIF file
    4. Saving with the original filename (without chunk suffixes)

    Parameters
    ----------
    input_dir : str
        Directory containing .pt files
    output_dir : str
        Directory to save .fif files

    Returns
    -------
    results : dict
        Dictionary with 'successful', 'failed', 'total', and 'errors' keys

    Examples
    --------
    >>> import zuna
    >>>
    >>> # Convert all PT files in directory
    >>> # Multiple PT files from same source will be stitched together
    >>> results = zuna.pt_directory_to_fif('data/pt_files', 'data/fif_files')
    >>> print(f"Converted {results['successful']} files")
    """
    from pathlib import Path
    from collections import defaultdict
    from .preprocessing.io import load_pt

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all PT files
    input_path = Path(input_dir)
    pt_files = list(input_path.glob("*.pt"))

    if len(pt_files) == 0:
        return {
            'successful': 0,
            'failed': 0,
            'total': 0,
            'errors': []
        }

    # Group PT files by original source filename
    source_groups = defaultdict(list)

    for pt_file in pt_files:
        try:
            # Load metadata to get original filename
            pt_data = load_pt(str(pt_file))
            metadata = pt_data.get('metadata', {})
            original_filename = metadata.get('original_filename')

            if original_filename:
                # Group by original filename
                source_groups[original_filename].append(pt_file)
            else:
                # Fallback: treat as standalone file
                source_groups[pt_file.name].append(pt_file)
        except Exception as e:
            # If we can't load metadata, treat as standalone
            source_groups[pt_file.name].append(pt_file)

    # Convert each group
    successful = 0
    failed = 0
    errors = []

    for original_filename, pt_file_group in source_groups.items():
        try:
            # Sort PT files to ensure correct order (by filename)
            pt_file_group = sorted(pt_file_group)

            # Convert all PT files to Raw objects
            raw_objects = []
            for pt_file in pt_file_group:
                raw = pt_to_raw(str(pt_file))
                raw_objects.append(raw)

            # Concatenate if multiple files
            if len(raw_objects) > 1:
                # Concatenate all raw objects
                combined_raw = mne.concatenate_raws(raw_objects, preload=True)
            else:
                combined_raw = raw_objects[0]

            # Generate output filename (remove chunk suffixes from original name)
            # Remove .fif extension if present, we'll add it back
            base_name = original_filename.replace('.fif', '').replace('.FIF', '')
            fif_name = base_name + ".fif"
            fif_path = output_path / fif_name

            # Save as FIF
            combined_raw.save(str(fif_path), overwrite=True)

            successful += 1

        except Exception as e:
            failed += 1
            errors.append({
                'files': [str(f) for f in pt_file_group],
                'original_filename': original_filename,
                'error': str(e)
            })

    return {
        'successful': successful,
        'failed': failed,
        'total': len(source_groups),
        'errors': errors
    }


# Expose key classes for advanced usage
from .preprocessing import ProcessingConfig, EEGProcessor
from .preprocessing.batch import process_directory
from .pipeline import run_zuna, zuna_preprocessing, zuna_inference, zuna_pt_to_fif

__all__ = [
    'raw_to_pt',
    'pt_to_raw',
    'pt_directory_to_fif',
    'process_directory',
    'run_zuna',
    'zuna_preprocessing',
    'zuna_inference',
    'zuna_pt_to_fif',
    'ProcessingConfig',
    'EEGProcessor',
]
