#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CALL SCRTIP LIKE THIS: 
systemd-run --user --scope -p MemoryMax=250G python 1_original_to_pt_with_location.py --quiet --dataset tuh
(has memory max)

OpenNeuro to .pt with 3D channel locations (Updated with batching)

What it does
------------
- Takes OpenNeuro data and converts directly to .pt files with 3D channel positions
- Keeps ALL available channels (no fixed 64-channel subset)
- Downsamples to 256Hz and epochs into 5-second segments
- Adds 3D spatial coordinates from standard 10-20 system for each channel
- DROPS channels without valid 3D coordinates and tracks them in 'channel_names_noloc'
- Stores data in format: {'data': tensor, 'channel_positions': coords, 'channel_names': names, 'channel_names_noloc': dropped}
- Uses batching mechanism for cross-file normalization (similar to 2_fif_to_pt.py)
- New filename format: dataset_batch1_batch2_d##_samples_channels_timepoints.pt

Key differences from original script 3
--------------------------------------
- Drops channels without valid 3D coordinates
- Adds 'channel_names_noloc' key to track dropped channels
- Uses batching mechanism for better normalization across multiple files
- New filename format with dataset name and batch indices

Usage
-----
$ python3 1_original_to_pt_with_location.py --dataset tuh
$ python3 1_original_to_pt_with_location.py --dataset one
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import glob
import mne
import numpy as np
import torch
import warnings
import json
import math
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
import signal
import sys
from tqdm import tqdm
from datetime import datetime
import argparse
from scipy.signal import welch
import gc
import psutil

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt, peak_widths

import matplotlib
import matplotlib.pyplot as plt
plt.ioff()
matplotlib.rcParams['figure.max_open_warning'] = 0



# Import plotting utilities
from eeg_plotting_utils_simple import plot_all_channels_comparison, plot_psd_focused_comparison
from eeg_plotting_utils_simple_psd import plot_all_channels_psd_comparison

QUIET = False
PLOTTING = True
PLOT_EVERY_N_FILES = 100

MAX_DURATION_MINUTES = 10
max_duration_seconds = MAX_DURATION_MINUTES * 60

warnings.filterwarnings("ignore")
mne.set_log_level('ERROR')

# Suppress MNE verbose output by setting all MNE functions to use verbose=False
import os
os.environ['MNE_LOGGING_LEVEL'] = 'ERROR'

# ---------- CONFIG ----------
# TUH vs OpenNeuro configuration - now set via command line argument

# GB constant for file size calculations
GB = 1024**3

def setup_config(dataset_type):
    """Setup configuration based on dataset type."""
    if dataset_type == 'tuh':
        return {
            'ROOT_PATH': '/mnt/raid0/bci/tuh/',
            'SAVE_DIR': "/mnt/shared/datasets/bci/pt3d/v5/tuh",
            'N_JOBS': 25,
            'FILES_PER_CHUNK': 1,
        }
    elif dataset_type == 'one':
        return {
            # 'ROOT_PATH': '/mnt/raid0/bci/openneuro/', #jm
            # 'SAVE_DIR': "/mnt/shared/datasets/bci/pt3d/v5/train/one", #jm
            'ROOT_PATH': '/data/datasets/bci/dataset_downloads_cw_BIG/', #jm
            'SAVE_DIR': "/data/datasets/bci/eval_chris_pt3d_BIG/", #jm
            # 'SAVE_DIR': "/mnt/raid0/bci_pt3d_2/one",
            'N_JOBS': 5,
            'FILES_PER_CHUNK': 1,
        }
    elif dataset_type == 'flat':
        return {
            'ROOT_PATH': '/data/datasets/bci/dataset_downloads_cw/jonas01/',
            'SAVE_DIR': "/data/datasets/bci/pt3d_eval_jonas01/",
            'N_JOBS': 5,
            'FILES_PER_CHUNK': 1,
        }
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Use 'tuh', 'one', or 'flat'")

# META_SUBDIR will be set in main() after config is loaded

SFREQ_FINAL = 256  # Target sampling rate
SAMPLES_PER_EPOCH = 5 * SFREQ_FINAL  # 5 seconds per epoch
MAX_EPOCHS_PER_PT = 64  # Fixed number of epochs per .pt file
MAX_CHUNK = MAX_EPOCHS_PER_PT  # Keep for compatibility
DTYPE = "float32"

# Filtering and cleaning params
OUTLIER_SD = 3
CLIP_Z = 5.0
HPF_FREQ = 0.5

# ============ TUNABLE SIGNAL QUALITY PARAMETERS ============
MAX_EPOCH_REJECTION_RATE = 0.15
BAD_EPOCH_AMPLITUDE_THRESHOLD = 10.0
BAD_CHANNEL_STD_MULTIPLIER = 5.0
VARIANCE_THRESHOLD_MULTIPLIER = 5.0

# Whether to enable more aggressive cleaning steps
ENABLE_AGGRESSIVE_CLEANING = False
# Whether to show verbose debug output during cleaning
DEBUG_VERBOSE = False

def collect_dataset_paths(config, dataset_type):
    root = config['ROOT_PATH']
    if dataset_type == 'one':
        # OpenNeuro: every direct subfolder is a dataset
        dataset_paths = []
        for dirpath, dirnames, _ in os.walk(root):
            if dirpath == root:
                for d in dirnames:
                    p = os.path.join(root, d)
                    if os.path.isdir(p):
                        dataset_paths.append(p)
        return sorted(dataset_paths)
    elif dataset_type == 'flat':
        # Flat: single directory with files
        return [root]
    else:
        # TUH: treat the whole tree as ONE dataset called "tuh"
        return [root]

def make_file_chunks(dataset_path, dataset_name, config, dataset_type):
    formats_and_loaders = get_formats_and_loaders(dataset_type)

    # 1) collect files
    print(f"📂 Scanning {dataset_path} for files...")
    dataset_files = []
    file_count = 0
    for dirpath, _, filenames in os.walk(dataset_path):
        for filename in filenames:
            for ext, loader in formats_and_loaders.items():
                if filename.lower().endswith(f".{ext}"):
                    file_path = os.path.join(dirpath, filename)
                    dataset_files.append((file_path, loader, dataset_name, ext))
                    file_count += 1
                    if file_count % 1000 == 0:
                        print(f"   Found {file_count} files so far...")

    print(f"✅ Found {len(dataset_files)} total files")

    if not dataset_files:
        return []

    # Special handling for flat: each file gets its own dataset_name extracted from filename
    if dataset_type == 'flat':
        import re
        chunks_with_names = []
        for idx, (file_path, loader, _, ext) in enumerate(dataset_files):
            filename = os.path.basename(file_path)
            # Extract number from filename like "eval_raw-20.fif" -> "20"
            match = re.search(r'eval_raw-(\d+)', filename)
            if match:
                file_num = match.group(1)
                file_dataset_name = f"ds{file_num}"
            else:
                file_dataset_name = f"ds{idx}"  # Fallback
            chunks_with_names.append((file_dataset_name, idx, [(file_path, loader, file_dataset_name, ext)]))
        return chunks_with_names

    files_per_chunk = config.get("FILES_PER_CHUNK", None)

    if files_per_chunk and files_per_chunk > 0:
        n_splits = math.ceil(len(dataset_files) / files_per_chunk)
        chunks = [
            dataset_files[i * files_per_chunk : (i + 1) * files_per_chunk]
            for i in range(n_splits)
        ]
    else:
        # fallback to old size-based logic
        total_bytes = sum(os.path.getsize(p[0]) for p in dataset_files if os.path.exists(p[0]))
        will_split = total_bytes >= config['SPLIT_THRESHOLD_GB'] * GB
        if will_split:
            n_splits = max(1, math.ceil(total_bytes / (config['CHUNK_TARGET_GB'] * GB)))
            n_splits = min(n_splits, len(dataset_files))
            chunks_np = np.array_split(np.array(dataset_files, dtype=object), n_splits)
            chunks = [chunk.tolist() for chunk in chunks_np]
        else:
            chunks = [dataset_files]

    # 3) return (dataset_name, chunk_idx, file_infos)
    return [
        (dataset_name, chunk_idx, chunk)
        for chunk_idx, chunk in enumerate(chunks)
        if len(chunk) > 0
    ]

# def chunk_already_done(save_dir, dataset_name, chunk_idx, expected_files=None):
#     """
#     Check if this logical chunk was already processed.

#     Rules:
#     - if there is at least one .pt that starts with {dataset}_{chunk}_ we *assume* it's done
#     - if expected_files is given, we can do a stricter check:
#         count how many .done_ files we have for this chunk
#         and only skip if count >= expected_files
#     """
#     # 1) old behavior (keep it!)
#     prefix = f"{dataset_name}_{chunk_idx:06d}_"
#     pattern_pt = os.path.join(save_dir, prefix + "*.pt")
#     pt_matches = glob.glob(pattern_pt)

#     if expected_files is None:
#         # fallback to original behaviour
#         return len(pt_matches) > 0

#     # 2) stricter: count .done_ markers
#     done_pattern = os.path.join(save_dir, f".done_{dataset_name}_{chunk_idx:06d}_*.json")
#     done_matches = glob.glob(done_pattern)

#     # if we have markers for all files, we can skip
#     if len(done_matches) >= expected_files:
#         return True

#     # else, not done
#     return False

def chunk_already_done(save_dir, dataset_name, chunk_idx, expected_files=None):
    """
    Consider a chunk 'done' only when all its source files have .done markers.
    If expected_files is None, never short-circuit at chunk level (return False)
    so we rely on per-file .done checks inside processing.
    """
    if expected_files is None:
        return False  # don't skip whole chunk unless we know how many files it has

    pattern = os.path.join(save_dir, f".done_{dataset_name}_{chunk_idx:06d}_*.json")
    done_matches = glob.glob(pattern)
    return len(done_matches) >= expected_files

def file_done_marker(save_dir, dataset_name=None, chunk_idx=None, base_name=None):
    """
    Return the path to the per-source-file done marker.

    Backward-compatible:
    - if only save_dir is given → return a generic marker
    - if all are given → return the real per-file marker
    """
    if dataset_name is None or chunk_idx is None or base_name is None:
        return os.path.join(save_dir, ".done_generic.json")

    return os.path.join(
        save_dir,
        f".done_{dataset_name}_{chunk_idx:06d}_{base_name}.json",
    )


def is_source_file_done(save_dir, dataset_name, chunk_idx, base_name):
    marker_path = file_done_marker(save_dir, dataset_name, chunk_idx, base_name)
    return os.path.exists(marker_path)


def mark_source_file_done(
    save_dir,
    dataset_name,
    chunk_idx,
    base_name,
    src_path,
    ok=True,
    error=None,
):
    marker_path = file_done_marker(save_dir, dataset_name, chunk_idx, base_name)
    with open(marker_path, "w") as f:
        json.dump(
            {
                "ok": ok,
                "dataset": dataset_name,
                "chunk_idx": chunk_idx,
                "base_name": base_name,
                "src": src_path,
                "ts": datetime.now().isoformat(),
                "error": error,
            },
            f,
        )

def get_formats_and_loaders(dataset_type):
    """Get file format loaders based on dataset type."""
    if dataset_type == 'tuh':
        # TUH only uses EDF files
        return {
            'edf': mne.io.read_raw_edf,
        }
    else:
        # OpenNeuro uses various formats
        return {
            'edf': mne.io.read_raw_edf,
            'vhdr': mne.io.read_raw_brainvision,
            'bdf': mne.io.read_raw_bdf,
            'fif': mne.io.read_raw_fif,
            'cnt': mne.io.read_raw_cnt,
            'mff': mne.io.read_raw_egi,
            'set': mne.io.read_raw_eeglab,
        }

def normalize_channel_name(ch, dataset_type):
    """Normalize channel names for consistent mapping."""
    ch = ch.lower().strip()
    
    if dataset_type == 'tuh':
        # TUH-specific channel cleaning (from script 2)
        ch = ch.replace("eeg ", "").replace("-ref", "").replace("-le", "").strip()
    else:
        # OpenNeuro channel cleaning
        for prefix in ['n-', 'eeg ', 'brainvision rda_', 'rda_', '1-', '2-']:
            if ch.startswith(prefix):
                ch = ch.replace(prefix, '')
    
    ch = ch.split('-')[0]
    ch = ch.rstrip('.')
    return ch

def load_biosemi_mapping(path):
    """Load biosemi mapping for channel renaming (from script 1)."""
    df = pd.read_csv(path)
    return {
        str(row['Electrode']).lower(): str(row['ch_name']).lower()
        for _, row in df.iterrows()
        if pd.notna(row['Electrode']) and pd.notna(row['ch_name'])
    }

def load_biosemi_mapping_with_coords(path):
    """Load biosemi mapping with both 10-20 names and 3D coordinates."""
    df = pd.read_csv(path)
    mapping = {}
    for _, row in df.iterrows():
        if pd.notna(row['Electrode']):
            electrode = str(row['Electrode']).lower()
            ch_name = str(row['ch_name']).lower() if pd.notna(row['ch_name']) else None
            x = float(row['x']) if pd.notna(row['x']) else 0.0
            y = float(row['y']) if pd.notna(row['y']) else 0.0  
            z = float(row['z']) if pd.notna(row['z']) else 0.0
            mapping[electrode] = {
                'ch_name': ch_name,
                'coords': np.array([x, y, z])
            }
    return mapping

# Load biosemi mappings for channel renaming
biosemi_maps = {
    128: load_biosemi_mapping("biosemi_chnames/biosemi128.csv"),
    160: load_biosemi_mapping("biosemi_chnames/biosemi160.csv"),
    256: load_biosemi_mapping("biosemi_chnames/biosemi256.csv"),
    '128_e': load_biosemi_mapping("biosemi_chnames/biosemi128_e.csv"),
    '160_e': load_biosemi_mapping("biosemi_chnames/biosemi160_e.csv"),
    '256_e': load_biosemi_mapping("biosemi_chnames/biosemi256_e.csv"),
}

# Load biosemi mappings with coordinates
biosemi_maps_with_coords = {
    128: load_biosemi_mapping_with_coords("biosemi_chnames/biosemi128.csv"),
    160: load_biosemi_mapping_with_coords("biosemi_chnames/biosemi160.csv"),
    256: load_biosemi_mapping_with_coords("biosemi_chnames/biosemi256.csv"),
    '128_e': load_biosemi_mapping_with_coords("biosemi_chnames/biosemi128_e.csv"),
    '160_e': load_biosemi_mapping_with_coords("biosemi_chnames/biosemi160_e.csv"),
    '256_e': load_biosemi_mapping_with_coords("biosemi_chnames/biosemi256_e.csv"),
}

def detect_biosemi_cap_type(channel_names, dataset_type):
    """Detect biosemi cap type from channel names."""
    ch_names_norm = [normalize_channel_name(ch, dataset_type) for ch in channel_names]
    normed = set(ch_names_norm)
    
    # Check for full biosemi layout (A+B+C series)
    if 'a1' in normed and 'b1' in normed and 'c1' in normed:
        cap_type = 128 if len(normed) <= 150 else 160 if len(normed) <= 190 else 256
        return cap_type, "biosemi_abc"
    # Check for E-series biosemi
    elif 'e1' in normed and 'e2' in normed and 'e3' in normed:
        cap_type = '128_e' if len(normed) <= 140 else '160_e' if len(normed) <= 180 else '256_e'
        return cap_type, "biosemi_e"
    # Check for partial biosemi (just A series or A+B series) - NEW!
    elif 'a1' in normed and any(f'a{i}' in normed for i in range(2, 10)):
        # Has multiple A-series channels, likely biosemi
        a_count = len([ch for ch in normed if ch.startswith('a') and ch[1:].isdigit()])
        b_count = len([ch for ch in normed if ch.startswith('b') and ch[1:].isdigit()])
        total_count = a_count + b_count + len([ch for ch in normed if ch.startswith('c') and ch[1:].isdigit()])
        
        if total_count >= 180:
            cap_type = 256
        elif total_count >= 150:
            cap_type = 160
        else:
            cap_type = 128
        return cap_type, "biosemi_partial"
    else:
        return None, "standard"

def detect_egi_cap_type(channel_names, dataset_type):
    """
    Lightweight EGI detector.
    Returns (preferred_montage_name, reason) or (None, None).
    """
    normed = [normalize_channel_name(ch, dataset_type) for ch in channel_names]

    # E-series: E1, E2, ...
    e_like = [ch for ch in normed if ch.startswith('e') and ch[1:].isdigit()]
    if len(e_like) >= 10:
        max_e = max(int(ch[1:]) for ch in e_like if ch[1:].isdigit())
        if max_e >= 400:
            return "GSN-HydroCel-512", "egi_e_series_512"
        elif max_e >= 250:
            return "GSN-HydroCel-257", "egi_e_series_257"
        elif max_e >= 200:
            return "GSN-HydroCel-256", "egi_e_series_256"
        elif max_e >= 120:
            # 129 is slightly more common in MNE than 128 for EGI
            return "GSN-HydroCel-129", "egi_e_series_129"
        elif max_e >= 64:
            return "GSN-HydroCel-65", "egi_e_series_65"
        else:
            return "GSN-HydroCel-64", "egi_e_series_64"

    # A-series: A1, A2, ...
    a_like = [ch for ch in normed if ch.startswith('a') and ch[1:].isdigit()]
    if len(a_like) >= 20:
        max_a = max(int(ch[1:]) for ch in a_like if ch[1:].isdigit())
        if max_a >= 200:
            return "GSN-HydroCel-256", "egi_a_series_256"
        elif max_a >= 120:
            return "GSN-HydroCel-129", "egi_a_series_129"
        else:
            return "GSN-HydroCel-65", "egi_a_series_65"

    return None, None


def load_montage_positions(montage_name, dataset_type):
    """Return {normalized_name -> xyz} for an MNE montage, or {} on failure."""
    try:
        m = mne.channels.make_standard_montage(montage_name)
        pos = m.get_positions()['ch_pos']
        out = {}
        for ch, xyz in pos.items():
            nn = normalize_channel_name(ch, dataset_type)
            out[nn] = np.array([xyz[0], xyz[1], xyz[2]], dtype=float)
        return out
    except Exception:
        return {}

def get_3d_channel_positions_with_mapping(channel_names, dataset_type):
    """
    Get 3D coordinates for channels using a robust, multi-montage strategy:
    1. Try MNE standard_1005 (denser than 1020)
    2. If looks like EGI → try a family of GSN-HydroCel montages
    3. Try BioSemi CSVs (your existing code)
    4. Try a fallback montage list (1010, biosemi64, easycap-*)
    5. Drop channels without valid coordinates
    """
    # --- detect biosemi (your old code) ---
    cap_type, mapping_type = detect_biosemi_cap_type(channel_names, dataset_type)

    # --- detect EGI early ---
    egi_main_montage, egi_reason = detect_egi_cap_type(channel_names, dataset_type)

    positions = []
    mapped_names = []
    original_names = []
    kept_indices = []
    dropped_channels = []
    biosemi_renamed_channels = []

    mapping_info = {
        'cap_type': cap_type,
        'mapping_type': mapping_type,
        'mne_mapped': 0,
        'biosemi_mapped': 0,
        'egi_mapped': 0,
        'dropped': 0,
        'egi_detected': egi_main_montage is not None,
        'egi_source': egi_reason,
        'fallback_hits': {},
    }

    # ---- 1) primary montage = standard_1005 (denser than 1020) ----
    primary_montage_positions = load_montage_positions("standard_1005", dataset_type)

    # ---- 2) prepare EGI variants if EGI was detected ----
    egi_montage_positions_list = []
    if egi_main_montage is not None:
        # order matters: most likely first
        egi_candidates = [
            egi_main_montage,
            "GSN-HydroCel-257",
            "GSN-HydroCel-256",
            "GSN-HydroCel-129",
            "GSN-HydroCel-128",
            "GSN-HydroCel-65",
            "GSN-HydroCel-64",
            "GSN-HydroCel-64_1",
            "GSN-HydroCel-32",
        ]
        seen = set()
        for mname in egi_candidates:
            if mname in seen:
                continue
            seen.add(mname)
            posdict = load_montage_positions(mname, dataset_type)
            if posdict:
                egi_montage_positions_list.append((mname, posdict))

    # ---- 3) load biosemi rename map (your old behavior) ----
    biosemi_rename_map = {}
    if cap_type and cap_type in biosemi_maps:
        biosemi_rename_map = biosemi_maps[cap_type]

    # ---- 4) fallback montages to try per-channel if everything else fails ----
    fallback_montages = [
        "standard_1010",
        "standard_1020",     # keep as extra fallback
        "biosemi64",
        "easycap-M10",
        "easycap-M1",
        "GSN-HydroCel-128",  # in case we mis-detected EGI
    ]
    fallback_positions_cache = {}

    for idx, ch_name in enumerate(channel_names):
        normalized_ch = normalize_channel_name(ch_name, dataset_type)
        found = False

        # backup biosemi rename
        if normalized_ch in biosemi_rename_map:
            biosemi_renamed_channels.append(biosemi_rename_map[normalized_ch])
        else:
            biosemi_renamed_channels.append(ch_name)

        # ---- A) try primary (standard_1005) ----
        if normalized_ch in primary_montage_positions:
            xyz = primary_montage_positions[normalized_ch]
            if not np.allclose(xyz, [0.0, 0.0, 0.0]):
                positions.append(xyz)
                mapped_names.append(ch_name)
                original_names.append(ch_name)
                kept_indices.append(idx)
                mapping_info['mne_mapped'] += 1
                continue  # done with this channel

        # ---- B) try EGI family (if detected) ----
        if egi_montage_positions_list:
            for mname, posdict in egi_montage_positions_list:
                if normalized_ch in posdict:
                    xyz = posdict[normalized_ch]
                    if not np.allclose(xyz, [0.0, 0.0, 0.0]):
                        positions.append(xyz)
                        mapped_names.append(ch_name)
                        original_names.append(ch_name)
                        kept_indices.append(idx)
                        mapping_info['egi_mapped'] += 1
                        mapping_info['fallback_hits'][mname] = mapping_info['fallback_hits'].get(mname, 0) + 1
                        found = True
                        break
            if found:
                continue

        # ---- C) try BioSemi CSV (your existing code) ----
        if cap_type and cap_type in biosemi_maps_with_coords:
            bio_map = biosemi_maps_with_coords[cap_type]
            if normalized_ch in bio_map:
                bio_info = bio_map[normalized_ch]
                if not np.allclose(bio_info['coords'], [0.0, 0.0, 0.0]):
                    xyz = bio_info['coords'] / 1000.0  # mm → m
                    positions.append(xyz)
                    mapped_names.append(ch_name)
                    original_names.append(ch_name)
                    kept_indices.append(idx)
                    mapping_info['biosemi_mapped'] += 1
                    continue

        # ---- D) try generic fallback montages ----
        hit_fallback = False
        for fb_name in fallback_montages:
            if fb_name not in fallback_positions_cache:
                fallback_positions_cache[fb_name] = load_montage_positions(fb_name, dataset_type)
            fb_posdict = fallback_positions_cache[fb_name]
            if normalized_ch in fb_posdict and not np.allclose(fb_posdict[normalized_ch], [0.0, 0.0, 0.0]):
                positions.append(fb_posdict[normalized_ch])
                mapped_names.append(ch_name)
                original_names.append(ch_name)
                kept_indices.append(idx)
                mapping_info['fallback_hits'][fb_name] = mapping_info['fallback_hits'].get(fb_name, 0) + 1
                hit_fallback = True
                break
        if hit_fallback:
            continue

        # ---- E) drop if we really can't map it ----
        dropped_channels.append(ch_name)
        mapping_info['dropped'] += 1

    return (
        np.array(positions),
        mapped_names,
        original_names,
        kept_indices,
        dropped_channels,
        mapping_info,
        biosemi_renamed_channels,
    )


def process_eeg_chunk(file_infos, dataset_name, chunk_idx, config, dataset_type, QUIET=False):
    """Process multiple EEG files together for cross-file normalization."""
    global _file_counter
    _file_counter = 0   # start counting files for this chunk
    _reset_epoch_cache()
    plot_done = False

    # Progress feedback
    if not QUIET:
        print(f"📁 Processing {dataset_name} chunk {chunk_idx} ({len(file_infos)} files)")

    all_epochs = []
    all_metadata = []
    
    # Process each file in the chunk
    for file_info in file_infos:
        file_path, loader, _, ext = file_info
        
        # Generate unique identifier for this file
        relative_path = os.path.relpath(file_path, config['ROOT_PATH'])
        relative_base = os.path.splitext(relative_path)[0]
        base_name = relative_base.replace(os.sep, '_')

        metadata = {
            "dataset": dataset_name,
            "original_file": file_path,
            "file_extension": ext,
            "base_name": base_name,
            "timestamp_processed": datetime.now().isoformat(),
            "status": "failed",
            "error_reason": "",
            "processing_stats": {
                "channels_dropped_duplicate": 0,
                "channels_dropped_collision": 0,
                "channels_dropped_tuh_keywords": 0,
                "channels_dropped_no_3d_coords": 0,
                "channels_dropped_flat": 0,
                "channels_dropped_clipped": 0,
                "channels_dropped_noisy": 0,
                "epochs_dropped_rejection": 0,
                "epochs_dropped_bad_channels": 0,
                "epochs_dropped_outlier_detection": 0,
                "epochs_dropped_amplitude": 0,
                "notch_peaks_detected": 0,
                "notch_frequencies": [],
                "bad_channels_final": [],
                "channel_dropout_stages": {},
            }
        }
        
        # Initialize fallback variables for exception handling
        current_metadata = metadata.copy()
        current_base_name = base_name

        try:
            # Load raw data
            raw = loader(file_path, preload=False, verbose=False)

            # Store original info
            metadata.update({
                "orig_sfreq": float(raw.info['sfreq']),
                "orig_n_channels": len(raw.ch_names),
                "orig_duration_sec": float(raw.n_times / raw.info['sfreq']),
                "orig_ch_names": raw.ch_names,
            })
            
            # Skip if too short
            if raw.n_times / raw.info['sfreq'] < 10:
                metadata["error_reason"] = "file too short (<10s)"
                if not QUIET:
                    print(f"⚠️ Skipped {base_name}: too short")
                mark_source_file_done(
                    config['SAVE_DIR'], dataset_name, chunk_idx, base_name, file_path
                )
                continue
            original_base_name = base_name


            if raw.times[-1] > max_duration_seconds:
                # print(f"📂 File is {raw.times[-1]/60:.1f} min, chunking into {MAX_DURATION_MINUTES}min segments")
                n_chunks = int(np.ceil(raw.times[-1] / max_duration_seconds))

                processing_tasks = []
                for sub_chunk_idx in range(n_chunks):
                    start_time = sub_chunk_idx * max_duration_seconds
                    end_time = min((sub_chunk_idx + 1) * max_duration_seconds, raw.times[-1])
                    chunk_base_name = f"{original_base_name}_chunk{sub_chunk_idx:02d}"

                    processing_tasks.append({
                        'raw': raw.copy().crop(tmin=start_time, tmax=end_time),
                        'base_name': chunk_base_name,
                        'metadata': metadata.copy(),
                        'chunk_info': {
                            "is_chunked": True,
                            "chunk_index": sub_chunk_idx,
                            "total_chunks": n_chunks,
                            "chunk_start_time": start_time,
                            "chunk_end_time": end_time,
                            "original_file": file_path
                        }
                    })

            else:
                # print(f"📂 File is {raw.times[-1]/60:.1f} min, processing as single segment")
                processing_tasks = [{
                    'raw': raw.copy(),
                    'base_name': base_name,
                    'metadata': metadata.copy(),
                    'chunk_info': None
                }]

            # Process each task (single loop handles both chunked and non-chunked)
            for task in processing_tasks:
                current_raw = task['raw']
                current_base_name = task['base_name']
                current_metadata = task['metadata']

                # Check if already processed
                if is_source_file_done(config['SAVE_DIR'], dataset_name, chunk_idx, current_base_name):
                    if not QUIET:
                        print(f"  ⏩ {current_base_name[:50]}... (already done)")
                    continue

                # Load data for this segment
                if not QUIET:
                    print(f"  🔄 {current_base_name[:50]}... (processing)")

                # Random chance for plotting (1 in PLOT_EVERY_N_FILES chance)
                import random
                should_plot = PLOTTING and (random.randint(1, PLOT_EVERY_N_FILES) == 1)

                current_raw.load_data()

                # Update metadata if chunked
                if task['chunk_info']:
                    current_metadata.update({
                        "chunk_info": task['chunk_info'],
                        "base_name": current_base_name,
                        "orig_duration_sec": float(task['chunk_info']['chunk_end_time'] - task['chunk_info']['chunk_start_time'])
                    })

                #########
                ### Main processing starts here ###
                #########
                try:
                    current_raw.pick("eeg")
                except:
                    if not QUIET:
                        print(f"Warning: No explicit EEG channels in {current_base_name}, keeping all channels")
                
                # Skip if no channels remaining
                if len(current_raw.ch_names) == 0:
                    current_metadata["error_reason"] = "no EEG channels found"
                    if not QUIET:
                        print(f"⚠️ Skipped {current_base_name}: no channels")
                    mark_source_file_done(
                        config['SAVE_DIR'], dataset_name, chunk_idx, current_base_name, file_path
                    )
                    continue
                
                #########################
                # Channel renaming
                #########################

                # Remove duplicate channels (when lowercased)
                ch_names_lower = [ch.lower() for ch in current_raw.ch_names]
                to_drop = []
                seen = set()
                for orig_ch, low_ch in zip(current_raw.ch_names, ch_names_lower):
                    if low_ch in seen:
                        to_drop.append(orig_ch)
                    else:
                        seen.add(low_ch)
                if to_drop:
                    current_raw.drop_channels(to_drop)
                    current_metadata["processing_stats"]["channels_dropped_duplicate"] = len(to_drop)
                    current_metadata["processing_stats"]["channel_dropout_stages"]["duplicate_removal"] = to_drop

                # Normalize channel names with collision handling
                try:
                    current_raw.rename_channels(lambda ch: normalize_channel_name(ch, dataset_type))
                except ValueError as e:
                    if "not unique" in str(e):
                        # Handle channel name collisions by adding suffixes
                        new_names = {}
                        name_counts = {}
                        collision_count = 0
                        for ch in current_raw.ch_names:
                            normalized = normalize_channel_name(ch, dataset_type)
                            if normalized in name_counts:
                                name_counts[normalized] += 1
                                new_names[ch] = f"{normalized}_{name_counts[normalized]}"
                                collision_count += 1
                            else:
                                name_counts[normalized] = 0
                                new_names[ch] = normalized
                        current_raw.rename_channels(new_names)
                        current_metadata["processing_stats"]["channels_dropped_collision"] = collision_count
                        current_metadata["processing_stats"]["channel_dropout_stages"]["name_collisions"] = collision_count
                        if not QUIET:
                            print(f"🔧 {dataset_name}: Resolved channel name collisions")
                    else:
                        raise e

                # TUH-specific channel filtering
                if dataset_type == 'tuh':
                    # Drop non-EEG channels by keyword (from script 2)
                    UNWANTED_KEYWORDS = ("EKG", "RESP", "EMG", "ECG", "ROC", "LOC", "AUX", "REOG", "LEOG")
                    channels_to_drop = [
                        ch for ch in current_raw.ch_names if any(bad.lower() in ch.lower() for bad in UNWANTED_KEYWORDS)
                    ]
                    if channels_to_drop:
                        current_raw.drop_channels(channels_to_drop)
                        current_metadata["processing_stats"]["channels_dropped_tuh_keywords"] = len(channels_to_drop)
                        current_metadata["processing_stats"]["channel_dropout_stages"]["tuh_keywords"] = channels_to_drop
                        # print(f"🧹 {dataset_name}: Dropped TUH non-EEG channels: {channels_to_drop}")
                
                # Resample if needed
                # current_raw.crop(tmin=0.0, tmax=20) #JM DEBUG
                if current_raw.info['sfreq'] != SFREQ_FINAL:
                    current_raw.resample(SFREQ_FINAL, verbose=False)

                
                # Get 3D channel positions - try MNE first, then biosemi, keep original names
                channel_positions, mapped_names, original_names, kept_indices, dropped_channels, mapping_info, biosemi_renamed_channels = get_3d_channel_positions_with_mapping(current_raw.ch_names, dataset_type)
                
                # Drop channels without 3D coordinates from the current_raw data
                if dropped_channels:
                    current_raw.drop_channels(dropped_channels)
                    current_metadata["processing_stats"]["channels_dropped_no_3d_coords"] = len(dropped_channels)
                    current_metadata["processing_stats"]["channel_dropout_stages"]["no_3d_coordinates"] = dropped_channels
                    if not QUIET:
                        print(f"🗑️ {dataset_name}: Dropped {len(dropped_channels)} channels without valid 3D coordinates: {dropped_channels[:10]}{'...' if len(dropped_channels) > 10 else ''}")
                
                current_metadata.update({
                    "final_n_channels": len(current_raw.ch_names),
                    "final_ch_names": current_raw.ch_names,
                    "final_sfreq": float(current_raw.info['sfreq']),
                    "mapping_info": mapping_info,
                    "mapped_channel_names": mapped_names,
                    "channel_names_noloc": dropped_channels,
                    "dropped_channels_count": len(dropped_channels)
                })

                
                #############################
                ### Normalize ###########
                #############################
                
                good_picks = mne.pick_types(current_raw.info, eeg=True, meg=False, eog=False, stim=False, exclude='bads')
                sfreq = current_raw.info['sfreq']
                assert len(good_picks) > 0, "No good EEG channels remaining."

                # Normalize using ONLY good channels for mean/std calculation
                good_data = current_raw.get_data(picks=good_picks)
                global_mean = good_data.mean()
                global_std = good_data.std()
                if global_std == 0:
                    raise RuntimeError("Global std is zero; data appears constant.")

                # Apply normalization to ALL channels (good + bad)
                all_picks = mne.pick_types(current_raw.info, eeg=True, meg=False, eog=False, stim=False, exclude=[])
                all_data = current_raw.get_data(picks=all_picks)
                data_z = (all_data - global_mean) / global_std
                current_raw._data[all_picks] = data_z
                

                #############################
                ### BACKUP of current_raw ###########
                #############################
                
                # I will use this data later to compare how this data looks to the clenaed one
                current_raw_original = current_raw.copy()
                current_raw_original.crop(tmax=60)

                #############################
                ### SET UP ######
                #############################

                # Set up plotting directory and initial plots
                if should_plot:
                    if not QUIET:
                        print(f"    📊 Generating plots (1 in {PLOT_EVERY_N_FILES} chance)")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    figures_dir = os.path.join(config['SAVE_DIR'], 'figures')
                    os.makedirs(figures_dir, exist_ok=True)
                    plot_dir = os.path.join(figures_dir, f"{timestamp}_{dataset_name}")

                    current_raw.plot_psd(show=False).canvas.figure.savefig(f'{plot_dir}_plot_01psd_start.png', dpi=300)
                    current_raw.plot(duration=50, n_channels=40, show=False).canvas.figure.savefig(f'{plot_dir}_plot_01_start.png', dpi=300)
                    plt.close('all')
                else:
                    plot_dir = None

                #############################
                ### CLEANING 1 ##############
                ### BAD CHANNEL DETECTION ###
                #############################

                bads = set()
                picks = mne.pick_types(current_raw.info, eeg=True, meg=False, eog=False, stim=False, exclude=[])
                data_current_raw = current_raw.get_data(picks=picks)
                all_stds = np.array([data_current_raw[i].std() for i in range(len(picks))])
                median_std = np.median(all_stds)
                mad_std = np.median(np.abs(all_stds - median_std))

                # Channels with extremely low variance (relative to others)
                flat_threshold = median_std - 3.0 * mad_std if mad_std > 0 else median_std * 0.01

                clipped_idx = []
                flat_channels = []
                clipped_channels = []

                for i in range(len(picks)):
                    x = data_current_raw[i]
                    ch_sd = x.std()
                    ch_name = current_raw.ch_names[picks[i]]

                    # Check if channel is essentially flat (relative to dataset)
                    if ch_sd < max(flat_threshold, median_std * 1e-6):
                        clipped_idx.append(i)
                        flat_channels.append(ch_name)
                        continue

                    # Check for clipping (samples stuck at min/max)
                    eps = 1e-3 * ch_sd  # near-rail tolerance (relative to channel's own std)
                    near_max = np.isclose(x, x.max(), atol=eps)
                    near_min = np.isclose(x, x.min(), atol=eps)
                    frac = (near_max | near_min).mean()
                    if frac > 0.005:  # >0.5% samples hugging min/max
                        clipped_idx.append(i)
                        clipped_channels.append(ch_name)

                for idx in clipped_idx:
                    bads.add(current_raw.ch_names[picks[idx]])

                # Track bad channel statistics
                current_metadata["processing_stats"]["channels_dropped_flat"] = len(flat_channels)
                current_metadata["processing_stats"]["channels_dropped_clipped"] = len(clipped_channels)
                current_metadata["processing_stats"]["channel_dropout_stages"]["flat_channels"] = flat_channels
                current_metadata["processing_stats"]["channel_dropout_stages"]["clipped_channels"] = clipped_channels       

                current_raw.info['bads'] = sorted(list(bads))
                if should_plot:
                    current_raw.plot_psd(show=False, exclude='bads').canvas.figure.savefig(f'{plot_dir}_plot_02psd_bad_removed.png', dpi=300)
                    current_raw.plot(duration=50, n_channels=40, show=False).canvas.figure.savefig(f'{plot_dir}_plot_02_bad_removed.png', dpi=300)
                    plt.close('all')

                #############################
                ### CLEANING 2 ##############
                ### filtering & reference ###
                #############################

                current_raw.filter(l_freq=0.5, h_freq=None, verbose=False)
                if should_plot:
                    current_raw.plot_psd(show=False, exclude='bads').canvas.figure.savefig(f'{plot_dir}_plot_03psd_filter.png', dpi=300)
                    current_raw.plot(duration=50, n_channels=40, show=False).canvas.figure.savefig(f'{plot_dir}_plot_03_filter.png', dpi=300)
                    plt.close('all')

                current_raw.set_eeg_reference('average', projection=False, verbose=False)
                if should_plot:
                    current_raw.plot_psd(show=False, exclude='bads').canvas.figure.savefig(f'{plot_dir}_plot_04psd_ref.png', dpi=300)
                    current_raw.plot(duration=50, n_channels=40, show=False).canvas.figure.savefig(f'{plot_dir}_plot_04_ref.png', dpi=300)
                    plt.close('all')

                #############################
                ### CLEANING 3 ##############
                ### NOTCH FILTERING    ###
                #############################

                fmin = 45.0
                fmax = min( sfreq/2 - 1.0, 250.0 )
                psd = current_raw.compute_psd(method='welch', fmin=fmin, fmax=fmax, picks=good_picks, n_fft=4096)
                freqs = psd.freqs
                P = 10*np.log10(np.median(psd.get_data(), axis=0))

                # --- Remove 1/f trend with a median filter (odd kernel in bins ~ few Hz) ---
                df = np.diff(freqs)[0]
                k = int(np.round(5.0/df)) | 1
                baseline = medfilt(P, kernel_size=max(k,3))
                resid = P - baseline

                # --- Simple SD-based peak detection ---
                resid_mean = np.mean(resid)
                resid_std = np.std(resid)
                height_thresh = resid_mean + 4.0 * resid_std  # Only peaks >4 standard deviations above mean

                peaks, props = find_peaks(resid, height=height_thresh, prominence=2.0, width=(1, int(np.round(2.0/df))))

                # Keep only narrow peaks (safety in Hz)
                w_hz = peak_widths(resid, peaks, rel_height=0.5)[0] * df
                keep = w_hz <= 2.0
                pk_freqs = freqs[peaks][keep]

                # Apply notch filter based on detected peaks
                notch_frequencies_applied = []
                if pk_freqs.size:
                    detected_freqs = np.unique(np.round(pk_freqs, 2))

                    # Ensure minimum 2Hz separation between notch frequencies
                    if len(detected_freqs) > 1:
                        filtered_freqs = [detected_freqs[0]]
                        for freq in detected_freqs[1:]:
                            if freq - filtered_freqs[-1] >= 2.0:  # 2Hz minimum separation
                                filtered_freqs.append(freq)
                        detected_freqs = np.array(filtered_freqs)

                    try:
                        current_raw.notch_filter(freqs=detected_freqs, picks=all_picks, filter_length='auto', phase='zero', verbose=False)
                        notch_frequencies_applied = detected_freqs.tolist()
                    except ValueError as e:
                        if "Stop bands are not sufficiently separated" in str(e):
                            # Fallback: just notch the first (strongest) peak
                            if not QUIET:
                                print(f"⚠️ Frequencies too close, using only primary peak: {detected_freqs[0]} Hz")
                            current_raw.notch_filter(freqs=[detected_freqs[0]], picks=all_picks, filter_length='auto', phase='zero', verbose=False)
                            notch_frequencies_applied = [detected_freqs[0]]
                        else:
                            raise e

                # Track notch filtering statistics
                current_metadata["processing_stats"]["notch_peaks_detected"] = len(pk_freqs) if pk_freqs.size else 0
                current_metadata["processing_stats"]["notch_frequencies"] = notch_frequencies_applied
    

                if should_plot:
                    current_raw.plot_psd(show=False, exclude='bads').canvas.figure.savefig(f'{plot_dir}_plot_05psd_notch.png', dpi=300)
                    current_raw.plot(duration=50, n_channels=40, show=False).canvas.figure.savefig(f'{plot_dir}_plot_05_notch.png', dpi=300)
                    plt.close('all')


                #############################
                ### REMOVE NOISY CHANNELS ##############
                ###    ###
                #############################

                normalized_data = current_raw.get_data(picks=good_picks)  # Get normalized data from good channels
                channel_stds = np.std(normalized_data, axis=1)  # STD per channel
                mean_std = np.mean(channel_stds)  # Mean STD across all channels
                noisy_threshold = 3.0 * mean_std
                noisy_channels = []
                for i, ch_std in enumerate(channel_stds):
                    if ch_std > noisy_threshold:
                        ch_name = current_raw.ch_names[good_picks[i]]
                        noisy_channels.append(ch_name)
                        bads.add(ch_name)
                # Update current_raw.info['bads'] with new noisy channels
                current_raw.info['bads'] = sorted(bads)

                # Track noisy channel statistics
                current_metadata["processing_stats"]["channels_dropped_noisy"] = len(noisy_channels)
                current_metadata["processing_stats"]["channel_dropout_stages"]["noisy_channels"] = noisy_channels
                current_metadata["processing_stats"]["bad_channels_final"] = sorted(list(bads))

                if should_plot:
                    current_raw.plot_psd(show=False, exclude='bads').canvas.figure.savefig(f'{plot_dir}_plot_06psd_badsremoved.png', dpi=300)
                    current_raw.plot(duration=50, n_channels=40, show=False).canvas.figure.savefig(f'{plot_dir}_plot_06_badsremoved.png', dpi=300)
                    plt.close('all')



                epochs = mne.make_fixed_length_epochs(current_raw, duration=5, preload=True, verbose=False)
                if should_plot:
                    epochs.plot_psd(show=False, exclude='bads').canvas.figure.savefig(f'{plot_dir}_plot_07psd_epochs.png', dpi=300)
                    epochs.plot(n_channels=40, show=False).canvas.figure.savefig(f'{plot_dir}_plot_07_epochs.png', dpi=300)
                    plt.close('all')

                epochs.apply_baseline((None, None))
                epoch_data = epochs.get_data()
        
                # epochs.plot_psd(show=False, exclude='bads').canvas.figure.savefig(f'{plot_dir}_plot_08psd_epochsrejected.png', dpi=300)
                # epochs.plot(n_channels=40, scalings=dict(eeg=1), show=False).canvas.figure.savefig(f'{plot_dir}_plot_08_epochsrejected.png', dpi=300)

                #############################
                ### zero out bad channels
                #############################

                for bad_ch in bads:
                    if bad_ch in epochs.ch_names:
                        ch_idx = epochs.ch_names.index(bad_ch)
                        epoch_data[:, ch_idx, :] = 0.0

                #############################
                ### use the data from above and detect each channel/epoch combination that has an SD of more than 5X that of the group mean, or less than 0.1 of the group mean 
                ### I then want to remove that data, I think best would be to just fill with all 0s for now (and then we drop it later )
                #############################

                std=np.std(epoch_data, axis=2)
                overall_mean_std = np.mean(std)
                overall_std_of_stds = np.std(std)
                high_threshold = overall_mean_std * 3.0  # 3x mean
                low_threshold = overall_mean_std * 0.1   # 0.1x mean
                
                # Find outlier channel/epoch combinations
                high_outliers = std > high_threshold
                low_outliers = std < low_threshold
                all_outliers = high_outliers | low_outliers

                # Count outliers
                n_high = np.sum(high_outliers)
                n_low = np.sum(low_outliers)
                n_total_outliers = np.sum(all_outliers)

                # print(f"Outlier detection: {n_high} high-std, {n_low} low-std, {n_total_outliers} total outliers")
                # print(f"Thresholds: low={low_threshold:.4f}, high={high_threshold:.4f}")

                epoch_data_cleaned = epoch_data.copy()

                mask_full = np.broadcast_to(all_outliers[..., None], epoch_data_cleaned.shape)
                epoch_data_cleaned[mask_full] = 0.0

                ##############################
                # ADD: Peak-to-peak detection #JM, maybe remove, is partially redundant with thing above
                ##############################
                peak_to_peak = np.max(epoch_data, axis=2) - np.min(epoch_data, axis=2)
                amplitude_threshold = 20.0
                amplitude_outliers = peak_to_peak > amplitude_threshold
                all_outliers_combined = all_outliers | amplitude_outliers
                mask_combined = np.broadcast_to(all_outliers_combined[..., None], epoch_data_cleaned.shape)
                epoch_data_cleaned[mask_combined] = 0.0

                # Count additional outliers found by amplitude check
                additional_amplitude_outliers = np.sum(amplitude_outliers & ~all_outliers)

                # Track outlier detection statistics
                current_metadata["processing_stats"]["epochs_dropped_outlier_detection"] = int(n_total_outliers)
                current_metadata["processing_stats"]["epochs_dropped_amplitude"] = int(additional_amplitude_outliers)
                # print(f"Peak-to-peak detection found {additional_outliers} additional outlier channel/epoch combinations")


                # Update the histogram plot with threshold lines
                if should_plot:
                    plt.close('all')
                    plt.hist(std.flatten(), bins=50, alpha=0.7)
                    plt.axvline(low_threshold, color='red', linestyle='--', label=f'Low threshold: {low_threshold:.4f}')
                    plt.axvline(high_threshold, color='red', linestyle='--', label=f'High threshold: {high_threshold:.4f}')
                    plt.axvline(overall_mean_std, color='green', linestyle='-', label=f'Mean: {overall_mean_std:.4f}')
                    plt.xlabel('Standard Deviation')
                    plt.ylabel('Frequency')
                    plt.title('Epoch/Channel SD Distribution with Outlier Thresholds')
                    plt.legend()
                    plt.savefig(f'{plot_dir}_plot_09_epochSDhist.png')
                    plt.close()

                ##############################
                ### Remove entire epoch if more than 50% of channels were zeroes out
                ##############################  
                channels_remaining_per_epoch = np.sum(~all_outliers, axis=1)
                total_channels_per_epoch = epoch_data.shape[1]
                fraction_good_channels = channels_remaining_per_epoch / total_channels_per_epoch
                bad_epochs_mask = fraction_good_channels < 0.5
                n_bad_epochs = np.sum(bad_epochs_mask)
                epoch_data_cleaned[bad_epochs_mask] = 0.0

                # Track bad epoch statistics
                current_metadata["processing_stats"]["epochs_dropped_bad_channels"] = int(n_bad_epochs)
                # print(f"Removed {n_bad_epochs} entire epochs with <50% good channels")

                ##############################
                ### FINAL NORMALIZATION ####
                ##############################

                # Get only non-zero data for normalization statistics
                non_zero_mask = epoch_data_cleaned != 0.0
                non_zero_data = epoch_data_cleaned[non_zero_mask]

                if len(non_zero_data) > 0:
                    final_mean = np.mean(non_zero_data)
                    final_std = np.std(non_zero_data)
                    if final_std > 0:
                        epoch_data_cleaned[non_zero_mask] = (non_zero_data - final_mean) / final_std
                        current_metadata["processing_stats"]["final_normalization"] = {
                            "mean_before": float(final_mean),
                            "std_before": float(final_std),
                            "non_zero_elements": int(len(non_zero_data)),
                            "total_elements": int(epoch_data_cleaned.size)
                        }

                #############################
                ### Plot pre post ###########
                #############################
                epochs_original = mne.make_fixed_length_epochs(current_raw_original, duration=5, preload=True, verbose=False)
                epoch_data_original = epochs_original.get_data()
                min_epochs = min(epoch_data_original.shape[0], epoch_data_cleaned.shape[0])

                epoch_data_original_aligned = epoch_data_original[:min_epochs, :, :]
                epoch_data_processed_aligned = epoch_data[:min_epochs, :, :]
                epoch_data_cleaned_aligned = epoch_data_cleaned[:min_epochs, :, :]

                # Use the updated plotting function with three datasets
                if should_plot:

                    # Normalize original data
                    original_mean = np.mean(epoch_data_original_aligned)
                    original_std = np.std(epoch_data_original_aligned)
                    if original_std > 0:
                        epoch_data_original_aligned = (epoch_data_original_aligned - original_mean) / original_std

                    # Normalize processed data  
                    processed_mean = np.mean(epoch_data_processed_aligned)
                    processed_std = np.std(epoch_data_processed_aligned)
                    if processed_std > 0:
                        epoch_data_processed_aligned = (epoch_data_processed_aligned - processed_mean) /processed_std

                    plot_all_channels_comparison(
                        data_before=epoch_data_original_aligned,
                        data_after=epoch_data_cleaned_aligned,
                        ch_names=epochs.ch_names,
                        sfreq=SFREQ_FINAL,
                        save_path=f'{plot_dir}_plot_10_three_stage_comparison.png',
                        n_plots_per_dataset=1,
                        removed_channels=list(bads),
                        data_middle=epoch_data_processed_aligned,
                        MAX_CHANNELS_FOR_GRID=256,
                        figsize=(30, 20)
                    )

                    plot_all_channels_psd_comparison(
                        data_before=epoch_data_original_aligned,
                        data_after=epoch_data_cleaned_aligned,
                        ch_names=epochs.ch_names,
                        sfreq=SFREQ_FINAL,
                        save_path=f'{plot_dir}_plot_11_three_stage_psd_comparison.png',
                        n_plots_per_dataset=1,
                        removed_channels=list(bads),
                        data_middle=epoch_data_processed_aligned,
                        MAX_CHANNELS_FOR_GRID=256,
                        figsize=(40, 30)
                    )

                    # Final plot cleanup
                    plt.close('all')
                    gc.collect()




                #############################
                ### Convert to list format and save ###
                #############################

                # Convert epoch_data_cleaned to list format, removing all-zero epochs/channels
                epochs_to_save = []
                positions_to_save = []

                for epoch_idx in range(epoch_data_cleaned.shape[0]):
                    epoch = epoch_data_cleaned[epoch_idx]  # Shape: (channels, samples)
                    non_zero_channels = ~np.all(epoch == 0, axis=1)

                    if np.sum(non_zero_channels) == 0:
                        continue
                    # Extract only non-zero channels
                    clean_epoch = epoch[non_zero_channels]
                    clean_positions = channel_positions[non_zero_channels]
                    epochs_to_save.append(clean_epoch)
                    positions_to_save.append(clean_positions)

                # print(f"Prepared {len(epochs_to_save)} epochs for saving")

                # Add final summary statistics
                total_epochs_original = epoch_data.shape[0] if 'epoch_data' in locals() else 0
                total_channels_start = len(current_metadata.get('orig_ch_names', []))
                total_channels_end = len(current_raw.ch_names)
                epochs_with_data = len(epochs_to_save)

                current_metadata["processing_stats"].update({
                    "summary": {
                        "total_epochs_original": total_epochs_original,
                        "total_epochs_saved": epochs_with_data,
                        "total_channels_start": total_channels_start,
                        "total_channels_end": total_channels_end,
                        "channel_retention_rate": total_channels_end / total_channels_start if total_channels_start > 0 else 0,
                        "epoch_retention_rate": epochs_with_data / total_epochs_original if total_epochs_original > 0 else 0,
                    }
                })

                # Use existing saving mechanism
                success_msg = _add_to_epoch_cache_and_save(
                    epochs_to_save,
                    [ch for ch, keep in zip(epochs.ch_names, ~np.all(epoch_data_cleaned == 0, axis=(0,2))) if keep],
                    channel_positions,
                    mapped_names,
                    original_names,
                    dropped_channels,
                    mapping_info,
                    current_metadata,
                    dataset_name,
                    chunk_idx,
                    current_base_name,
                    config
                )

                mark_source_file_done(
                    config['SAVE_DIR'],
                    dataset_name,
                    chunk_idx,
                    current_base_name,
                    file_path,
                    ok=True,
                    error=None,
                )

                if not QUIET:
                    print(f"  ✅ {current_base_name[:50]}... (completed)")

                # Comprehensive memory cleanup after each file
                del current_raw, epochs, epoch_data, epoch_data_cleaned
                if 'epoch_data_original' in locals():
                    del epoch_data_original
                if 'epochs_original' in locals():
                    del epochs_original
                if 'current_raw_original' in locals():
                    del current_raw_original
                plt.close('all')
                gc.collect()



        except Exception as e:
            current_metadata["error_reason"] = f"Exception: {str(e)}"
            if not QUIET:
                print(f"  ❌ {current_base_name[:50]}... (failed: {str(e)[:50]})")
            mark_source_file_done(
                config['SAVE_DIR'],
                dataset_name,
                chunk_idx,
                current_base_name,
                file_path,
                ok=False,
                error=str(e),
            )
            continue               

    if not QUIET:
        print(f"🎯 Completed {dataset_name} chunk {chunk_idx}")
    return f"✅ {dataset_name} chunk {chunk_idx}: processed files individually"
    



_file_counter = 0
_global_file_counter = 0
_global_processing_counter = 0  # Tracks total files processed for plotting frequency
_epoch_cache = {
    'data_list': [],
    'channel_positions_list': [],
    'metadata_cache': [],
    'reference_info': None  # Store channel info from first cached epoch
}


def _reset_epoch_cache():
    """Reset the global epoch cache to prevent memory accumulation."""
    global _epoch_cache
    _epoch_cache['data_list'].clear()
    _epoch_cache['channel_positions_list'].clear()
    _epoch_cache['metadata_cache'].clear()
    _epoch_cache['reference_info'] = None
    gc.collect()

def _add_to_epoch_cache_and_save(data_cleaned, cleaned_ch_names, channel_positions, mapped_names,
                                original_names, dropped_channels, mapping_info, metadata,
                                dataset_name, chunk_idx, base_name, config):
    """Add epochs to cache and save when we reach MAX_EPOCHS_PER_PT."""
    global _epoch_cache, _file_counter

    # Handle different data formats from cleaning
    if isinstance(data_cleaned, list):
        # 5-step normalization format
        epochs_to_add = data_cleaned
        positions_to_add = [channel_positions[:epoch_data.shape[0]] for epoch_data in data_cleaned]
    else:
        # Standard format
        epochs_to_add = [data_cleaned[i] for i in range(data_cleaned.shape[0])]
        positions_to_add = [channel_positions for _ in range(len(epochs_to_add))]

    if len(epochs_to_add) == 0:
        return f"⚠️ Skipped {base_name}: no epochs after cleaning"

    # Store reference info from first file
    if _epoch_cache['reference_info'] is None:
        _epoch_cache['reference_info'] = {
            'cleaned_ch_names': cleaned_ch_names,
            'mapped_names': mapped_names,
            'original_names': original_names,
            'dropped_channels': dropped_channels,
            'mapping_info': mapping_info,
            'dataset_name': dataset_name,
            'chunk_idx': chunk_idx,
            'config': config
        }

    # Add epochs to cache
    _epoch_cache['data_list'].extend(epochs_to_add)
    _epoch_cache['channel_positions_list'].extend(positions_to_add)
    _epoch_cache['metadata_cache'].extend([metadata] * len(epochs_to_add))

    saved_files = []
    epochs_added = len(epochs_to_add)

    # Save complete PT files when we have enough epochs
    while len(_epoch_cache['data_list']) >= MAX_EPOCHS_PER_PT:
        saved_file = _save_pt_from_cache()
        saved_files.append(saved_file)

    total_cached = len(_epoch_cache['data_list'])
    if saved_files:
        return f"💾 Added {epochs_added} epochs, saved {len(saved_files)} PT files, {total_cached} epochs cached"
    else:
        return f"📦 Added {epochs_added} epochs to cache, total cached: {total_cached}"


def _save_pt_from_cache():
    """Save one PT file from the current cache."""
    global _file_counter, _epoch_cache

    if len(_epoch_cache['data_list']) < MAX_EPOCHS_PER_PT:
        return "⚠️ Not enough epochs in cache to save"

    # Extract epochs for this PT file
    epochs_for_pt = _epoch_cache['data_list'][:MAX_EPOCHS_PER_PT]
    positions_for_pt = _epoch_cache['channel_positions_list'][:MAX_EPOCHS_PER_PT]
    metadata_for_pt = _epoch_cache['metadata_cache'][:MAX_EPOCHS_PER_PT]

    # Remove from cache
    _epoch_cache['data_list'] = _epoch_cache['data_list'][MAX_EPOCHS_PER_PT:]
    _epoch_cache['channel_positions_list'] = _epoch_cache['channel_positions_list'][MAX_EPOCHS_PER_PT:]
    _epoch_cache['metadata_cache'] = _epoch_cache['metadata_cache'][MAX_EPOCHS_PER_PT:]

    # Use reference info
    ref = _epoch_cache['reference_info']

    # Generate filename
    _file_counter += 1
    n_dropped = len(ref['dropped_channels'])
    avg_channels = sum(epoch.shape[0] for epoch in epochs_for_pt) / len(epochs_for_pt)
    filename = f"{ref['dataset_name']}_{ref['chunk_idx']:06d}_{_file_counter:06d}_d{n_dropped:02d}_{MAX_EPOCHS_PER_PT:05d}_{int(avg_channels)}_{SAMPLES_PER_EPOCH}.pt"

    # Convert to tensors
    data_tensors = [torch.tensor(epoch_data, dtype=torch.float32) for epoch_data in epochs_for_pt]
    position_tensors = [torch.tensor(epoch_pos, dtype=torch.float32) for epoch_pos in positions_for_pt]

    # Create data dictionary
    data_dict = {
        'data': data_tensors,
        'channel_positions': position_tensors,
        'metadata': {
            'original_sampling_rate': float(metadata_for_pt[0].get('orig_sfreq', 256.0)),
            'resampled_sampling_rate': float(SFREQ_FINAL),
            'channel_names_10_20': ref['cleaned_ch_names'],
            'channel_names_original': ref['original_names'],
            'channel_names_mapped': ref['mapped_names'],
            'channel_names_noloc': ref['dropped_channels'],
            'mapping_info': ref['mapping_info'],
            'dataset_info': {
                'dataset': ref['dataset_name'],
                'chunk_index': ref['chunk_idx'],
                'file_counter': _file_counter,
                'epochs_per_file': MAX_EPOCHS_PER_PT,
                'source_files': [m.get('original_file', '') for m in metadata_for_pt[:5]]  # First 5 source files
            },
            'processing_stats': {
                'n_epochs': MAX_EPOCHS_PER_PT,
                'avg_channels_per_epoch': float(avg_channels),
                'final_data_structure': "list_of_2d_arrays",
                'timestamp_processed': datetime.now().isoformat(),
                'status': "processed"
            }
        }
    }

    # Save PT file directly (no .tmp backup)
    pt_path = os.path.join(ref['config']['SAVE_DIR'], filename)
    torch.save(data_dict, pt_path)

    return f"💾 Saved {filename}: {MAX_EPOCHS_PER_PT} epochs, ~{avg_channels:.1f} channels"

def _flush_remaining_cache():
    """Save any remaining epochs in cache at the end of processing."""
    global _epoch_cache, _file_counter

    # if we never added any epoch, there is nothing to flush
    if _epoch_cache['reference_info'] is None:
        return "📦 No reference info, nothing to flush"

    if len(_epoch_cache['data_list']) == 0:
        return "📦 No remaining epochs to flush"

    # Save remaining epochs (even if < MAX_EPOCHS_PER_PT)
    epochs_for_pt = _epoch_cache['data_list']
    positions_for_pt = _epoch_cache['channel_positions_list']
    metadata_for_pt = _epoch_cache['metadata_cache']
    n_remaining = len(epochs_for_pt)

    # clear cache
    _epoch_cache['data_list'] = []
    _epoch_cache['channel_positions_list'] = []
    _epoch_cache['metadata_cache'] = []

    ref = _epoch_cache['reference_info']

    _file_counter += 1
    n_dropped = len(ref['dropped_channels'])
    avg_channels = sum(epoch.shape[0] for epoch in epochs_for_pt) / len(epochs_for_pt)
    filename = (
        f"{ref['dataset_name']}_{ref['chunk_idx']:06d}_{_file_counter:06d}"
        f"_d{n_dropped:02d}_{n_remaining:05d}_{int(avg_channels)}_{SAMPLES_PER_EPOCH}.pt"
    )

    data_tensors = [torch.tensor(ep, dtype=torch.float32) for ep in epochs_for_pt]
    position_tensors = [torch.tensor(pos, dtype=torch.float32) for pos in positions_for_pt]

    data_dict = {
        "data": data_tensors,
        "channel_positions": position_tensors,
        "metadata": {
            "original_sampling_rate": float(metadata_for_pt[0].get("orig_sfreq", 256.0)),
            "resampled_sampling_rate": float(SFREQ_FINAL),
            "channel_names_10_20": ref["cleaned_ch_names"],
            "channel_names_original": ref["original_names"],
            "channel_names_mapped": ref["mapped_names"],
            "channel_names_noloc": ref["dropped_channels"],
            "mapping_info": ref["mapping_info"],
            "dataset_info": {
                "dataset": ref["dataset_name"],
                "chunk_index": ref["chunk_idx"],
                "file_counter": _file_counter,
                "epochs_per_file": n_remaining,
                "source_files": [m.get("original_file", "") for m in metadata_for_pt[:5]],
            },
            "processing_stats": {
                "n_epochs": n_remaining,
                "avg_channels_per_epoch": float(avg_channels),
                "final_data_structure": "list_of_2d_arrays",
                "timestamp_processed": datetime.now().isoformat(),
                "status": "processed_final",
            },
        },
    }

    pt_path = os.path.join(ref["config"]["SAVE_DIR"], filename)
    torch.save(data_dict, pt_path)

    return f"💾 Flushed remaining {n_remaining} epochs to {filename}"

def main():
    global QUIET
    parser = argparse.ArgumentParser(description="Convert TUH / OpenNeuro EEG to .pt with 3D coords and fixed-size epoch packs")
    parser.add_argument('--dataset', required=True, choices=['tuh', 'one', 'flat'], help='Which dataset layout to use')
    parser.add_argument('--quiet', action='store_true', help='less verbose')
    args = parser.parse_args()
    QUIET = args.quiet
    config = setup_config(args.dataset)
    dataset_type = args.dataset

    os.makedirs(config['SAVE_DIR'], exist_ok=True)

    # 1) get dataset roots
    dataset_paths = collect_dataset_paths(config, dataset_type)
    # print(f"Found {len(dataset_paths)} dataset roots to process.")

    # 2) flatten into work items
    planned_work_items = []
    for dataset_path in dataset_paths:
        dataset_name = Path(dataset_path).name if dataset_type == 'one' else "ds000000"
        chunks = make_file_chunks(dataset_path, dataset_name, config, dataset_type)
        planned_work_items.extend(chunks)

    # print(f"Built {len(planned_work_items)} chunk tasks (before skip).")

    # 3) FILTER: keep only those chunks that don't already have their first file (FAST VERSION)
    print(f"🔍 Checking {len(planned_work_items)} chunks for existing work...")

    # Pre-load all existing .done_ files for fast lookup
    print("   📋 Building skip cache...")
    done_files = set()
    done_pattern = os.path.join(config['SAVE_DIR'], ".done_*.json")
    for done_file in glob.glob(done_pattern):
        done_files.add(os.path.basename(done_file))
    print(f"   Found {len(done_files)} existing done markers")

    # Now check quickly using the cache
    work_items = []
    skipped_count = 0
    checked_count = 0

    for (dataset_name, chunk_idx, file_infos) in planned_work_items:

        if chunk_already_done(
                config['SAVE_DIR'],
                dataset_name,
                chunk_idx,
                expected_files=len(file_infos)
            ):
            skipped_count += 1
            continue

        # if chunk_already_done(config['SAVE_DIR'], dataset_name, chunk_idx):
        #     skipped_count += 1
        #     continue
        
        checked_count += 1
        if checked_count % 5000 == 0:  # Less frequent updates since it's faster now
            print(f"   Checked {checked_count}/{len(planned_work_items)} chunks, skipped {skipped_count} so far...")

        # Fast lookup using pre-built set
        if len(file_infos) > 0:
            # Check if first file in chunk is done (simplified check)
            file_path, _, _, ext = file_infos[0]
            relative_path = os.path.relpath(file_path, config['ROOT_PATH'])
            relative_base = os.path.splitext(relative_path)[0]
            base_name = relative_base.replace(os.sep, '_')
            expected_done_file = f".done_{dataset_name}_{chunk_idx:06d}_{base_name}.json"

            if expected_done_file in done_files:
                skipped_count += 1
                continue

        work_items.append((dataset_name, chunk_idx, file_infos))

    # Show skip summary
    if skipped_count > 0:
        print(f"⏩ Skipped {skipped_count} chunks already processed")
    if len(work_items) > 0:
        print(f"🚀 Processing {len(work_items)} chunks...")

    if not work_items:
        # print("No work to do.")
        # still flush cache just in case
        # print("\n💾 Flushing remaining epochs from cache...")
        final_save_result = _flush_remaining_cache()
        if final_save_result:
            print(final_save_result)
        return

    # 4) run in parallel only the remaining ones
    results = []
    try:
        with Parallel(
            n_jobs=config['N_JOBS'],
            backend="loky",
            verbose=0,
            timeout=10800,
            max_nbytes=None,
        ) as parallel:
            results = parallel(
                delayed(process_eeg_chunk)(file_infos, dataset_name, chunk_idx, config, dataset_type, QUIET)
                for (dataset_name, chunk_idx, file_infos) in tqdm(work_items, desc="Processing chunks")
            )
    except KeyboardInterrupt:
        # print("⚠️ Caught KeyboardInterrupt, shutting down workers…")
        # make sure loky’s global executor is also nuked
        get_reusable_executor().shutdown(wait=True)
        raise
    finally:
        # extra safety: make *sure* loky global pool is gone
        try:
            get_reusable_executor().shutdown(wait=True)
        except Exception:
            pass

    # Print results
    success_count = 0
    skip_count = 0
    error_count = 0

    for result in results:
        if isinstance(result, list):
            for sub_result in result:
                # print(sub_result)
                if "✅" in sub_result:
                    success_count += 1
                elif "⏩" in sub_result:
                    skip_count += 1
                elif "❌" in sub_result or "⚠️" in sub_result:
                    error_count += 1
        else:
            # print(result)
            if "✅" in result:
                success_count += 1
            elif "⏩" in result:
                skip_count += 1
            elif "❌" in result or "⚠️" in result:
                error_count += 1

    # 6) final flush
    # print("\n💾 Flushing remaining epochs from cache...")
    final_save_result = _flush_remaining_cache()
    if final_save_result:
        # print(final_save_result)
        pass

    # Show completion summary
    print(f"✅✅✅✅✅ Processing complete - Output: {config['SAVE_DIR']}✅✅✅✅✅")

if __name__ == "__main__":
    main()