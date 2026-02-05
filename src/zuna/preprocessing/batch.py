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


def _generate_output_filename(
    dataset_name: str,
    file_idx: int,
    metadata: Dict[str, Any],
    epochs_list: List,
) -> str:
    """
    Generate output filename in format:
    {dataset_name}_000000_{file_idx:06d}_d{n_dropped:02d}_{n_epochs:05d}_{avg_channels}_{samples_per_epoch}.pt

    Example: ds000123_000000_000001_d05_00064_063_1280.pt
    """
    chunk_idx = 0  # Single chunk for now
    n_dropped = len(metadata.get('channels_dropped_no_coords', []))
    n_epochs = len(epochs_list)
    avg_channels = int(np.mean([ep.shape[0] for ep in epochs_list])) if epochs_list else 0
    samples_per_epoch = epochs_list[0].shape[1] if epochs_list else 0

    filename = (
        f"{dataset_name}_{chunk_idx:06d}_{file_idx:06d}_"
        f"d{n_dropped:02d}_{n_epochs:05d}_{avg_channels}_{samples_per_epoch}.pt"
    )
    return filename


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

        # Check if file needs chunking
        max_duration_seconds = config.max_duration_minutes * 60
        file_duration = raw.times[-1]

        chunks_to_process = []

        if file_duration > max_duration_seconds:
            # Split into chunks
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

        # Process each chunk
        chunk_results = []
        for chunk_info in chunks_to_process:
            chunk_raw = chunk_info['raw']

            # Process
            epochs_list, positions_list, metadata = processor.process(chunk_raw)

            if len(epochs_list) == 0:
                chunk_results.append({
                    'success': False,
                    'error': 'No epochs after processing'
                })
                continue

            # Generate dataset name
            if chunk_info['is_chunked']:
                dataset_name = f"ds{file_counter:06d}_chunk{chunk_info['chunk_idx']:02d}"
            else:
                dataset_name = f"ds{file_counter:06d}"

            # Generate output filename
            output_filename = _generate_output_filename(
                dataset_name=dataset_name,
                file_idx=idx,
                metadata=metadata,
                epochs_list=epochs_list
            )
            output_file = output_path / output_filename

            # Save
            from .io import save_pt
            save_pt(
                epochs_list,
                positions_list,
                metadata['channel_names'],
                str(output_file),
                metadata=metadata,
                reversibility_params=metadata.get('reversibility')
            )

            chunk_results.append({
                'success': True,
                'epochs_saved': metadata['n_epochs_saved'],
                'output': str(output_file),
                'metadata': metadata,
                'chunk_idx': chunk_info['chunk_idx']
            })

            # Memory cleanup
            del chunk_raw, epochs_list, positions_list
            gc.collect()

        # Return summary
        successful_chunks = [r for r in chunk_results if r['success']]
        if successful_chunks:
            return {
                'file': file_path.name,
                'success': True,
                'file_counter': file_counter,
                'chunks': len(chunk_results),
                'successful_chunks': len(successful_chunks),
                'outputs': [r['output'] for r in successful_chunks],
                'total_epochs': sum(r.get('epochs_saved', 0) for r in successful_chunks)
            }
        else:
            return {
                'file': file_path.name,
                'success': False,
                'file_counter': file_counter,
                'error': 'No chunks processed successfully'
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

    Output files will be named: ds{file_num:06d}_{chunk:06d}_{file:06d}_d{dropped:02d}_{epochs:05d}_{channels}_{samples}.pt
    Example: ds000001_000000_000001_d05_00064_063_1280.pt
    Chunked files: ds000001_chunk00_000000_000001_d05_00064_063_1280.pt

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
                if 'chunks' in result:
                    print(f"     Chunks: {result['successful_chunks']}/{result['chunks']}")
                    print(f"     Total epochs: {result['total_epochs']}")
                    print(f"     Outputs: {len(result['outputs'])} files")
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

    # Final summary
    print("\n" + "="*80)
    print("Processing Summary")
    print("="*80)
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    total_epochs = sum(r.get('total_epochs', r.get('epochs_saved', 0)) for r in results if r['success'])
    total_output_files = sum(len(r.get('outputs', [])) for r in results if r['success'])

    print(f"  Total input files: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total output PT files: {total_output_files}")
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
