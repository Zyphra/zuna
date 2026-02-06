#!/bin/bash

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# This script is used to evaluate the BCI model on different training runs and checkpoints.
# It loops through predefined training runs and their associated parameters, and for each run,
# it loops through a set of checkpoints to evaluate the model performance and make plots of samples.
#
# 1st, setup tmux and docker with lingua.sh
#   >> bash /mnt/home/chris/workspace/AY2l/lingua/lingua.sh (on Crusoe)
#
# 2nd, run something like:
#   >> bash /mnt/home/chris/workspace/AY2l/lingua/apps/AY2latent_bci/loop_eeg_eval.sh
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Define training runs and params to be defined for each run
#                   ONe, CR1     ONe, CR8    TUH, CR8    TUH, CR4    TUH, CR2     TUH, CR1     
training_runs=(     "test31"     "test30"    "test28"    "test27"    "test26"    "test21")
sample_rates=(      "512"         "512"       "256"       "256"       "256"       "256")
seq_lens=(          "2560"        "2560"      "1280"      "1280"      "1280"      "1280")
data_norms=(        "5"           "5"         "1"         "1"         "1"         "1") 
data_dirs=(         "/workspace/bci/data/mmap_june16_padded_fp32_HPF_chunked2_eval"
                    "/workspace/bci/data/mmap_june16_padded_fp32_HPF_chunked2_eval"
                    "/workspace/bci/data/tmp_tuh_eval"
                    "/workspace/bci/data/tmp_tuh_eval"
                    "/workspace/bci/data/tmp_tuh_eval"
                    "/workspace/bci/data/tmp_tuh_eval")

input_dims=(        "64"         "64"        "23"        "23"        "23"        "23")
encoder_in_dims=(   "64"         "64"        "23"        "23"        "23"        "23")
encoder_out_dims=(  "128"        "64"        "23"        "23"        "23"        "46")
encoder_latent_dss=( "2"         "8"         "8"         "4"         "2"         "2")

sliding_windows=(   "128"        "256"      "256"       "256"       "128"       "128" ) 
encoder_windows=(   "128"        "256"      "256"       "256"       "128"       "128" ) 

# Define the inner loop over checkpoints (NOTE: If checkpoint isnt there, it fails gracefully and moves onto next one.)
checkpoints=("0000010000" "0000020000" "0000030000" "0000040000" "0000050000" "0000060000" "0000070000" "0000080000" "0000090000" "0000100000" \
             "0000110000" "0000120000" "0000130000" "0000140000" "0000150000" "0000160000" "0000170000" "0000180000" "0000190000" "0000200000" \
             "0000210000" "0000220000" "0000230000" "0000240000" "0000250000" "0000260000" "0000270000" "0000280000" "0000290000" "0000300000" \
             "0000310000" "0000320000" "0000330000" "0000340000" "0000350000")

num_items=${#training_runs[@]}

# Outer loop over training runs and associated config params.
for (( i=0; i<$num_items; i++ )); do

    # Assign variables for the current training run and its parameters
    train="${training_runs[$i]}"
    fs="${sample_rates[$i]}"
    num_t="${seq_lens[$i]}"
    data_dir="${data_dirs[$i]}"
    data_norm="${data_norms[$i]}"
    input_dim="${input_dims[$i]}"
    encoder_input_dim="${input_dims[$i]}"
    encoder_output_dim="${encoder_out_dims[$i]}"
    encoder_latent_downsample_factor="${encoder_latent_dss[$i]}"
    sliding_window="${sliding_windows[$i]}"
    encoder_sliding_window="${encoder_windows[$i]}"

    # Inner loop over checkpoints
    for ckpt in "${checkpoints[@]}"; do
        init_ckpt_path="/workspace/bci/checkpoints/bci/bci_AY2l_"$train"/checkpoints/"$ckpt""
        output_save_dir="/workspace/AY2l/lingua/figures/bci_AY2l_"$train"/checkpoints/"$ckpt"/cfg1.0"

        echo "  ***** init_ckpt_path: $init_ckpt_path - fs: $fs - num_t: $num_t - data_dir: $data_dir - data_norm: $data_norm - input_dim: $input_dim - encoder_input_dim: $encoder_input_dim - encoder_output_dim: $encoder_output_dim - encoder_latent_downsample_factor: $encoder_latent_downsample_factor - sliding_window: $sliding_window - encoder_sliding_window: $encoder_sliding_window"
    
        # Check if output directory exists and contains files already - dont rerun if so. 
        if [ -d "$output_save_dir" ] && [ "$(ls -A $output_save_dir)" ]; then
            echo "  Output directory $output_save_dir already exists and is not empty. Skipping this run."
            continue
        fi

        # If MSE plots dont exist already, run the evaluation script.
        CUDA_VISIBLE_DEVICES=5 python3 apps/AY2latent_bci/eeg_eval.py config=apps/AY2latent_bci/configs/config_bci_eval.yaml \
            data.sample_rate=$fs \
            data.seq_len=$num_t \
            data.data_dir=$data_dir \
            data.data_norm=$data_norm \
            model.input_dim=$input_dim \
            model.encoder_input_dim=$encoder_input_dim \
            model.encoder_output_dim=$encoder_output_dim \
            model.encoder_latent_downsample_factor=$encoder_latent_downsample_factor \
            model.sliding_window=$sliding_window \
            model.encoder_sliding_window=$encoder_sliding_window \
            checkpoint.init_ckpt_path=$init_ckpt_path
    
    done
done
