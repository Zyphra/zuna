#!/usr/bin/env python3
"""
Tutorial: Zuna Preprocessing Pipeline
"""
import zuna
from zuna.preprocessing import process_directory


CONFIG = {
    'drop_bad_channels': False,
    'drop_bad_epochs': False,
    'apply_notch_filter': False,
    'upsample_to_channels': 32,  # Upsample to 128 channels (set to None to disable)
}

INPUT_DIR = "data/1_fif_input"
OUTPUT_DIR = "data/2_pt_input"
N_JOBS = 10  # Use 1 for single file, >1 for parallel processing

results = process_directory(INPUT_DIR, OUTPUT_DIR, n_jobs=N_JOBS, **CONFIG)

successful = [r for r in results if r['success']]
failed = [r for r in results if not r['success']]
print(f"Processed {len(successful)} files successfully")

