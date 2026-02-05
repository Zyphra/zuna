"""
Batch processing utilities for multiple EEG files.
"""
import mne
from pathlib import Path
from typing import List, Dict, Any, Optional
from .processor import EEGProcessor
from .config import ProcessingConfig


def process_directory(
    input_dir: str,
    output_dir: str,
    config: Optional[ProcessingConfig] = None,
    montage_name: Optional[str] = "standard_1005",
    **config_kwargs
) -> List[Dict[str, Any]]:
    """
    Process all EEG files in a directory.

    Automatically detects file formats and processes each file,
    saving PT files to output directory.

    Parameters
    ----------
    input_dir : str
        Path to directory containing raw EEG files (.fif, .edf, etc.)
    output_dir : str
        Path to directory where PT files will be saved
    config : ProcessingConfig, optional
        Configuration object. If None, uses defaults.
    montage_name : str, optional
        Name of MNE standard montage to use if file doesn't have one.
        Set to None to require files to already have montages.
        Default: "standard_1005"
    **config_kwargs
        Additional config parameters (e.g., drop_bad_channels=True)

    Returns
    -------
    results : list of dict
        List of processing results for each file

    Examples
    --------
    >>> import zuna
    >>> results = zuna.preprocessing.process_directory(
    ...     "/data/raw_files",
    ...     "/data/processed_pt",
    ...     drop_bad_channels=True
    ... )
    >>> successful = sum(1 for r in results if r['success'])
    >>> print(f"Processed {successful}/{len(results)} files")
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all supported EEG files
    supported_extensions = ['.fif', '.edf', '.bdf', '.vhdr', '.cnt', '.set', '.mff']
    eeg_files = []
    for ext in supported_extensions:
        eeg_files.extend(input_path.glob(f'**/*{ext}'))

    if len(eeg_files) == 0:
        print(f"⚠️  No EEG files found in {input_dir}")
        print(f"   Looking for: {', '.join(supported_extensions)}")
        return []

    print("="*80)
    print(f"Found {len(eeg_files)} EEG file(s) to process")
    print("="*80)

    # Load montage once if specified
    montage = None
    if montage_name is not None:
        montage = mne.channels.make_standard_montage(montage_name)

    # Create config
    if config is None:
        config = ProcessingConfig(**config_kwargs)
    elif config_kwargs:
        # Merge kwargs into config
        config_dict = config.__dict__.copy()
        config_dict.update(config_kwargs)
        config = ProcessingConfig(**config_dict)

    # Create processor
    processor = EEGProcessor(config)

    # Process each file
    results = []
    for idx, file_path in enumerate(eeg_files, 1):
        print(f"\n[{idx}/{len(eeg_files)}] Processing: {file_path.name}")

        try:
            # Load raw data - auto-detect file type
            raw = _load_raw_file(file_path)

            # Check if montage already exists
            if raw.get_montage() is None:
                if montage is None:
                    print(f"  ⚠️  Skipping: No montage in file and no montage_name provided")
                    results.append({
                        'file': file_path.name,
                        'success': False,
                        'error': 'No montage available'
                    })
                    continue
                # Set montage
                raw.set_montage(montage, match_case=False, on_missing='ignore')
            else:
                print(f"  ✓ Using existing montage from file")

            # Generate output filename
            output_file = output_path / f"{file_path.stem}.pt"

            # Process
            metadata = processor.process_and_save(raw, str(output_file))

            # Print summary
            print(f"  ✅ Success!")
            print(f"     Epochs: {metadata['n_epochs_original']} → {metadata['n_epochs_saved']}")
            print(f"     Channels: {metadata['original_n_channels']} → {metadata['final_n_channels']}")
            print(f"     Bad channels: {len(metadata['bad_channels'])}")
            print(f"     Output: {output_file}")

            results.append({
                'file': file_path.name,
                'success': True,
                'epochs_saved': metadata['n_epochs_saved'],
                'output': str(output_file),
                'metadata': metadata
            })

        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results.append({
                'file': file_path.name,
                'success': False,
                'error': str(e)
            })

    # Final summary
    print("\n" + "="*80)
    print("Processing Summary")
    print("="*80)
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    total_epochs = sum(r.get('epochs_saved', 0) for r in results if r['success'])

    print(f"  Total files: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total epochs saved: {total_epochs}")
    print(f"  Output directory: {output_dir}")

    return results


def _load_raw_file(file_path: Path) -> mne.io.Raw:
    """
    Load raw EEG file, auto-detecting format.

    Parameters
    ----------
    file_path : Path
        Path to EEG file

    Returns
    -------
    raw : mne.io.Raw
        Loaded raw data
    """
    suffix = file_path.suffix.lower()

    loaders = {
        '.fif': mne.io.read_raw_fif,
        '.edf': mne.io.read_raw_edf,
        '.bdf': mne.io.read_raw_bdf,
        '.vhdr': mne.io.read_raw_brainvision,
        '.cnt': mne.io.read_raw_cnt,
        '.set': mne.io.read_raw_eeglab,
        '.mff': mne.io.read_raw_egi,
    }

    if suffix not in loaders:
        raise ValueError(f"Unsupported file format: {suffix}")

    loader = loaders[suffix]
    return loader(file_path, preload=True, verbose=False)
