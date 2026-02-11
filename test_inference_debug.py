#!/usr/bin/env python3
"""Quick test to see model inference debug output"""

from zuna import run_model_inference

# Run inference on just the PT input files
run_model_inference(
    input_dir='/data/datasets/bci/dataset_downloads_cw/pip_test/2_pt_input',
    output_dir='/data/datasets/bci/dataset_downloads_cw/pip_test/3_pt_output_debug',
    model_path='/data/weights/AY2l_model-weights/5c3mz37w/checkpoints/epoch=29-step=93930.ckpt',
    device='cuda:0'
)

print("\nDone! Check the output above for [DEBUG] messages")
