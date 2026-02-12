"""
Batch processing utilities for multiple EEG files.
"""
import mne
import re
import numpy as np
import gc
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
from joblib import Parallel, delayed
from .processor import EEGProcessor
from .config import ProcessingConfig


# Global epoch cache for batching epochs into 64-sample PT files
_epoch_cache = {
    'data_list': [],
    'positions_list': [],
    'channel_names': None,
    'metadata': None,
    'file_counter': 0,
    'pt_file_counter': 0  # Resets for each new source file
}


def _reset_epoch_cache():
    """Reset the global epoch cache."""
    global _epoch_cache
    _epoch_cache['data_list'].clear()
    _epoch_cache['positions_list'].clear()
    _epoch_cache['channel_names'] = None
    _epoch_cache['metadata'] = None
    _epoch_cache['pt_file_counter'] = 0
    gc.collect()


def _generate_output_filename(
    dataset_name: str,
    file_counter: int,
    pt_file_idx: int,
    n_epochs: int,
    metadata: Dict[str, Any],
    epochs_list: List,
) -> str:
    """
    Generate output filename in format:
    {dataset_name}_{file_counter:06d}_{pt_file_idx:06d}_d{n_dropped:02d}_{n_epochs:05d}_{avg_channels}_{samples_per_epoch}.pt

    Example: ds000000_000000_000001_d05_00064_063_1280.pt
             ds000000_000000_000002_d05_00064_063_1280.pt  (same source file)
             ds000000_000001_000001_d05_00064_063_1280.pt  (next source file)

    Where:
      - file_counter: Which source .fif file (0, 1, 2, ...)
      - pt_file_idx: Which PT file from that source (1, 2, 3, ...)
    """
    n_dropped = len(metadata.get('channels_dropped_no_coords', []))
    avg_channels = int(np.mean([ep.shape[0] for ep in epochs_list])) if epochs_list else 0
    samples_per_epoch = epochs_list[0].shape[1] if epochs_list else 0

    filename = (
        f"{dataset_name}_{file_counter:06d}_{pt_file_idx:06d}_"
        f"d{n_dropped:02d}_{n_epochs:05d}_{avg_channels}_{samples_per_epoch}.pt"
    )
    return filename


def _add_epochs_to_cache(
    epochs_list: List,
    positions_list: List,
    metadata: Dict[str, Any],
    file_counter: int,
    output_path: Path,
    config: ProcessingConfig
) -> List[str]:
    """
    Add epochs to cache and save PT files when we reach 64 epochs.

    Returns list of saved PT filenames.
    """
    global _epoch_cache

    # Check if we're starting a new source file
    if _epoch_cache['file_counter'] != file_counter:
        # Reset PT file counter for new source file
        _epoch_cache['pt_file_counter'] = 0
        _epoch_cache['file_counter'] = file_counter

    # Store metadata from first batch
    if _epoch_cache['metadata'] is None:
        _epoch_cache['metadata'] = metadata.copy()
        _epoch_cache['channel_names'] = metadata['channel_names']

    # Add epochs to cache
    _epoch_cache['data_list'].extend(epochs_list)
    _epoch_cache['positions_list'].extend(positions_list)

    saved_files = []

    #JM save pt - Save complete PT files (64 epochs each) when cache is full
    while len(_epoch_cache['data_list']) >= config.epochs_per_file:
        output_file = _save_pt_from_cache(output_path, config)
        if output_file:
            saved_files.append(output_file)

    return saved_files


def _save_pt_from_cache(output_path: Path, config: ProcessingConfig) -> Optional[str]:
    """Save one PT file (64 epochs) from the cache."""
    global _epoch_cache

    if len(_epoch_cache['data_list']) < config.epochs_per_file:
        return None

    # Extract epochs for this PT file
    epochs_for_pt = _epoch_cache['data_list'][:config.epochs_per_file]
    positions_for_pt = _epoch_cache['positions_list'][:config.epochs_per_file]

    # Remove from cache
    _epoch_cache['data_list'] = _epoch_cache['data_list'][config.epochs_per_file:]
    _epoch_cache['positions_list'] = _epoch_cache['positions_list'][config.epochs_per_file:]

    # Increment PT file counter
    _epoch_cache['pt_file_counter'] += 1

    # Generate filename
    dataset_name = "ds000000"  # Always use ds000000 as base
    output_filename = _generate_output_filename(
        dataset_name=dataset_name,
        file_counter=_epoch_cache['file_counter'],
        pt_file_idx=_epoch_cache['pt_file_counter'],
        n_epochs=config.epochs_per_file,
        metadata=_epoch_cache['metadata'],
        epochs_list=epochs_for_pt
    )
    output_file = output_path / output_filename

    #JM save pt - Call save_pt to write this batch of epochs to disk
    from .io import save_pt
    save_pt(
        epochs_for_pt,
        positions_for_pt,
        _epoch_cache['channel_names'],
        str(output_file),
        metadata=_epoch_cache['metadata'],
        reversibility_params=_epoch_cache['metadata'].get('reversibility')
    )

    return str(output_file)


def _flush_remaining_cache(output_path: Path) -> Optional[str]:
    """Save any remaining epochs in cache (< 64) at the end of processing."""
    global _epoch_cache

    if len(_epoch_cache['data_list']) == 0:
        return None

    if _epoch_cache['metadata'] is None:
        return None

    # Get remaining epochs and save metadata BEFORE clearing cache
    epochs_for_pt = _epoch_cache['data_list']
    positions_for_pt = _epoch_cache['positions_list']
    n_remaining = len(epochs_for_pt)

    # Increment PT file counter FIRST (before saving its value)
    _epoch_cache['pt_file_counter'] += 1

    # Save metadata and channel_names to local variables before resetting
    saved_metadata = _epoch_cache['metadata'].copy() if _epoch_cache['metadata'] else {}
    saved_channel_names = _epoch_cache['channel_names']
    saved_file_counter = _epoch_cache['file_counter']
    saved_pt_file_counter = _epoch_cache['pt_file_counter']

    # Clear cache
    _epoch_cache['data_list'] = []
    _epoch_cache['positions_list'] = []
    _epoch_cache['metadata'] = None  # Reset metadata to prevent carrying over to next file
    _epoch_cache['channel_names'] = None  # Reset channel names too

    # Generate filename using saved values
    dataset_name = "ds000000"  # Always use ds000000 as base
    output_filename = _generate_output_filename(
        dataset_name=dataset_name,
        file_counter=saved_file_counter,
        pt_file_idx=saved_pt_file_counter,
        n_epochs=n_remaining,
        metadata=saved_metadata,
        epochs_list=epochs_for_pt
    )
    output_file = output_path / output_filename

    #JM save pt - Flush remaining epochs (< 64) to final PT file
    from .io import save_pt
    save_pt(
        epochs_for_pt,
        positions_for_pt,
        saved_channel_names,
        str(output_file),
        metadata=saved_metadata,
        reversibility_params=saved_metadata.get('reversibility')
    )

    return str(output_file)


def _process_single_file(
    file_path: Path,
    idx: int,
    file_counter: int,
    output_path: Path,
    processor: EEGProcessor,
    config: ProcessingConfig,
) -> Dict[str, Any]:

    """
    Process a single EEG file (internal helper for parallel processing).

    Returns a dict with processing results.
    """
    try:
        # Load raw data - auto-detect file type
        raw = _load_raw_file(file_path)

        # Check if file has montage (REQUIRED)
        if raw.get_montage() is None:
            return {
                'file': file_path.name,
                'success': False,
                'error': 'No montage in file',
                'file_counter': file_counter
            }

        # Check if file needs chunking (for processing/normalization)
        # NOTE: With max_duration_minutes = 999999, chunking is effectively disabled
        # and preprocessed FIF will be saved correctly in processor.py
        max_duration_seconds = config.max_duration_minutes * 60
        file_duration = raw.times[-1]

        chunks_to_process = []

        if file_duration > max_duration_seconds:
            # Split into chunks for processing
            n_chunks = int(np.ceil(file_duration / max_duration_seconds))
            for sub_chunk_idx in range(n_chunks):
                start_time = sub_chunk_idx * max_duration_seconds
                end_time = min((sub_chunk_idx + 1) * max_duration_seconds, file_duration)
                chunks_to_process.append({
                    'raw': raw.copy().crop(tmin=start_time, tmax=end_time),
                    'chunk_idx': sub_chunk_idx,
                    'is_chunked': True,
                    'total_chunks': n_chunks
                })
        else:
            # Process as single file
            chunks_to_process.append({
                'raw': raw.copy(),
                'chunk_idx': None,
                'is_chunked': False,
                'total_chunks': 1
            })

        # Process each chunk and accumulate epochs
        total_epochs_from_file = 0
        saved_pt_files = []

        for chunk_info in chunks_to_process:
            chunk_raw = chunk_info['raw']

            # Process
            epochs_list, positions_list, metadata = processor.process(chunk_raw)

            # Add original filename to metadata for reconstruction
            metadata['original_filename'] = file_path.name

            if len(epochs_list) == 0:
                continue

            # Add epochs to cache (will save PT files when reaching 64 epochs)
            pt_files = _add_epochs_to_cache(
                epochs_list,
                positions_list,
                metadata,
                file_counter,
                output_path,
                config
            )

            saved_pt_files.extend(pt_files)
            total_epochs_from_file += len(epochs_list)

            # Memory cleanup
            del chunk_raw, epochs_list, positions_list
            gc.collect()

        # IMPORTANT: Flush cache after each source file to prevent mixing epochs
        # This ensures PT files only contain epochs from a single source file
        remaining_file = _flush_remaining_cache(output_path)
        if remaining_file:
            saved_pt_files.append(remaining_file)

        # Return summary
        if total_epochs_from_file > 0:
            return {
                'file': file_path.name,
                'success': True,
                'file_counter': file_counter,
                'chunks': len(chunks_to_process),
                'total_epochs': total_epochs_from_file,
                'pt_files_saved': len(saved_pt_files),
                'outputs': saved_pt_files
            }
        else:
            return {
                'file': file_path.name,
                'success': False,
                'file_counter': file_counter,
                'error': 'No epochs after processing'
            }

    except Exception as e:
        return {
            'file': file_path.name,
            'success': False,
            'file_counter': file_counter,
            'error': str(e)
        }



def process_directory(
    input_dir: str,
    output_dir: str,
    config: Optional[ProcessingConfig] = None,
    n_jobs: int = 1,
    **config_kwargs
) -> List[Dict[str, Any]]:
    """
    Process all EEG files in a directory with optional parallel processing.

    Automatically detects file formats and processes each file,
    saving PT files to output directory.

    **IMPORTANT**: All input files must already have montages set with 3D channel positions.
    Files without montages will be skipped.

    Features:
    - Parallel processing: Set n_jobs > 1 to process multiple files simultaneously
    - Automatic chunking: Files longer than max_duration_minutes are split into segments
    - Epoch batching: Accumulates epochs across files and saves in batches of 64

    Output files will be named: ds{file_num:06d}_{pt_file:06d}_d{dropped:02d}_{epochs:05d}_{channels}_{samples}.pt
    Example: ds000000_000001_d05_00064_063_1280.pt

    Files are numbered sequentially starting from 0.

    Parameters
    ----------
    input_dir : str
        Path to directory containing raw EEG files (.fif, .edf, etc.)
        Files must have montages already set.
    output_dir : str
        Path to directory where PT files will be saved
    config : ProcessingConfig, optional
        Configuration object. If None, uses defaults.
    n_jobs : int, optional
        Number of parallel jobs (default: 1 for sequential processing).
        Set to -1 to use all available CPUs.
    **config_kwargs
        Additional config parameters (e.g., drop_bad_channels=True, max_duration_minutes=10)

    Returns
    -------
    results : list of dict
        List of processing results for each file

    Examples
    --------
    >>> import zuna
    >>> # Sequential processing
    >>> results = zuna.preprocessing.process_directory(
    ...     "/data/raw_files",
    ...     "/data/processed_pt",
    ...     drop_bad_channels=True
    ... )
    >>>
    >>> # Parallel processing with 4 workers
    >>> results = zuna.preprocessing.process_directory(
    ...     "/data/raw_files",
    ...     "/data/processed_pt",
    ...     n_jobs=4,
    ...     max_duration_minutes=10
    ... )
    >>> successful = sum(1 for r in results if r['success'])
    >>> print(f"Processed {successful}/{len(results)} files")
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Reset epoch cache at the start
    _reset_epoch_cache()

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

    # Create config
    if config is None:
        config = ProcessingConfig(**config_kwargs)
    elif config_kwargs:
        # Merge kwargs into config
        config_dict = config.__dict__.copy()
        config_dict.update(config_kwargs)
        config = ProcessingConfig(**config_dict)

    # Create processor (one per job if parallel)
    processor = EEGProcessor(config)

    # Prepare file processing tasks
    file_counter = 0  # Running counter for dataset names, starts at 0
    tasks = []
    for idx, file_path in enumerate(eeg_files, 1):
        tasks.append((file_path, idx, file_counter))
        file_counter += 1

    # Process files (parallel or sequential)
    if n_jobs == 1:
        # Sequential processing with progress feedback
        results = []
        for file_path, idx, fc in tasks:
            print(f"\n[{idx}/{len(eeg_files)}] Processing: {file_path.name}")
            result = _process_single_file(file_path, idx, fc, output_path, processor, config)

            # Print summary
            if result['success']:
                print(f"  ✅ Success!")
                print(f"     Total epochs: {result['total_epochs']}")
                if result.get('pt_files_saved', 0) > 0:
                    print(f"     PT files saved: {result['pt_files_saved']}")
            else:
                print(f"  ⚠️  Skipped: {result.get('error', 'Unknown error')}")

            results.append(result)
    else:
        # Parallel processing
        print(f"\n🚀 Processing {len(tasks)} files with {n_jobs} parallel workers...")

        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
            delayed(_process_single_file)(
                file_path, idx, fc, output_path, processor, config
            )
            for file_path, idx, fc in tasks
        )

    # Flush remaining epochs in cache
    print("\n💾 Saving remaining epochs from cache...")
    remaining_file = _flush_remaining_cache(output_path)
    if remaining_file:
        print(f"  ✅ Saved remaining epochs to: {Path(remaining_file).name}")

    # Final summary
    print("\n" + "="*80)
    print("Processing Summary")
    print("="*80)
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    total_epochs = sum(r.get('total_epochs', 0) for r in results if r['success'])
    total_pt_files = sum(r.get('pt_files_saved', 0) for r in results if r['success'])

    # Add the final flushed file if it exists
    if remaining_file:
        total_pt_files += 1

    print(f"  Total input files: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total epochs processed: {total_epochs}")
    print(f"  Total PT files saved: {total_pt_files}")
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
