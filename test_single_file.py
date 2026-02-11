#!/usr/bin/env python3
"""Test inference on a single small file"""

import sys
sys.path.insert(0, '/data/home/jonas/workspace/zuna/src')

from zuna.pipeline import run_zuna

# Run on just the small file
run_zuna(
    input_dir='/data/datasets/bci/dataset_downloads_cw/pip_test/0_fif_input_small',
    output_dir='/data/datasets/bci/dataset_downloads_cw/pip_test/test_output',
    pt_input_dir='/data/datasets/bci/dataset_downloads_cw/pip_test/test_pt_in',
    pt_output_dir='/data/datasets/bci/dataset_downloads_cw/pip_test/test_pt_out',
    model_path='/data/weights/AY2l_model-weights/5c3mz37w/checkpoints/epoch=29-step=93930.ckpt',
    skip_preprocessing=False,
    skip_inference=False,
    skip_reconstruction=False
)
