# This script is running eeg_eval_jm_2.py
# And loops over different test datasets as well as over different checkpoints 
# TO RUN: 
# CUDA_VISIBLE_DEVICES=6 python3 apps/AY2latent_bci/eeg_eval_jm_2_loop.py

import subprocess
import csv
from pathlib import Path
import time

DATASET = "tuh"  # or "moabb"

if DATASET == "moabb":
    # Define your checkpoint paths and data dirs
    # MOABB
    checkpoints = [
        f"/workspace/bci/checkpoints/bci/bci_AY2l_test17/checkpoints/0000{i:02d}0000"
        for i in range(1, 14, 1)
    ]

    data_dirs = [
        "/workspace/bci/data/eval_datasets/eval_datasets_openneuro/moabb2016/data/",
        "/workspace/bci/data/eval_datasets/eval_datasets_openneuro/moabb2015/data/",
        "/workspace/bci/data/eval_datasets/eval_datasets_openneuro/moabb2017/data/",
        "/workspace/bci/data/eval_datasets/eval_datasets_openneuro/faced/data/",
        "/workspace/bci/data/eval_datasets/eval_datasets_openneuro/dreamer/data/",
    ]

if DATASET == "tuh":
    # TUH
    checkpoints = [
        f"/workspace/bci/checkpoints/bci/bci_AY2l_test22/checkpoints/0000{i:02d}0000"
        for i in range(1, 15, 1)
    ]

    data_dirs = [
        "/workspace/bci/data/eval_datasets/eval_datasets_tuh/moabb2016/data/",
        "/workspace/bci/data/eval_datasets/eval_datasets_tuh/moabb2015/data/",
        "/workspace/bci/data/eval_datasets/eval_datasets_tuh/moabb2017/data/",
        "/workspace/bci/data/eval_datasets/eval_datasets_tuh/faced/data/",
        "/workspace/bci/data/eval_datasets/eval_datasets_tuh/dreamer/data/",
    ]    



# Base config file
base_config = "apps/AY2latent_bci/configs/config_bci_eval_moabb.yaml"

# Output CSV path
output_path = Path("/workspace/bci/data/eval_datasets/eval_results_tuh_test22.csv")

if not output_path.exists():
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "checkpoint", "data_dir",
            "acc_in", "std_in",
            "acc_out", "std_out",
            "acc_random", "std_random"
        ])

total_runs = len(checkpoints) * len(data_dirs)
run_counter = 0

# Loop over all combinations
for data_dir in data_dirs:    
    for ckpt in checkpoints:        
        start_time = time.time()
        run_counter += 1
        print(f"\n[{run_counter}/{total_runs}] Running eval for ckpt={ckpt} and data_dir={data_dir}")

        cmd = [
            "python3",
            "apps/AY2latent_bci/eeg_eval_jm_2.py",
            f"config={base_config}",
            f"checkpoint.init_ckpt_path={ckpt}",
            f"data.data_dir={data_dir}",
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            out = result.stdout
        except subprocess.CalledProcessError as e:
            print(f"❌ Error for ckpt={ckpt} and data_dir={data_dir}")
            print("STDERR:\n", e.stderr)
            continue  # Skip to the next combo


        # Parse accuracies
        acc_in = std_in = acc_out = std_out = acc_random = std_random = None

        for line in out.splitlines():
            if "Accuracy (features from raw input)" in line:
                parts = line.strip().split()
                acc_in = float(parts[-3])
                std_in = float(parts[-1])
            if "Accuracy (features from autoencoder)" in line:
                parts = line.strip().split()
                acc_out = float(parts[-3])
                std_out = float(parts[-1])
            if "Accuracy (random baseline)" in line:
                parts = line.strip().split()
                acc_random = float(parts[-3])
                std_random = float(parts[-1])


        print(f"  -> acc_in={acc_in}, acc_out={acc_out}, acc_random={acc_random}")
        with open(output_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ckpt,
                data_dir,
                acc_in, std_in,
                acc_out, std_out,
                acc_random, std_random
            ])

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"⏱️  Time taken: {elapsed:.1f} seconds\n")