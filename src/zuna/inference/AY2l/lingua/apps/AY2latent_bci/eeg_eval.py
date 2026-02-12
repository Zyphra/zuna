
# 1st, 
#   >> pip install zuna

# 2nd, run something like:
#   >> CUDA_VISIBLE_DEVICES=3 python3 src/zuna/inference/AY2l/lingua/apps/AY2latent_bci/eeg_eval.py config=src/zuna/inference/AY2l/lingua/apps/AY2latent_bci/configs/config_bci_eval.yaml


import gc
import logging
import os
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import random
import numpy as np
from omegaconf import OmegaConf
import torch
import matplotlib.pyplot as plt

import torch.nn.attention.flex_attention
torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True

# To load model from HuggingFace.
import json
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load


from lingua.args import dataclass_from_dict, dump_config
from utils_pt_mne import interpolate_signals_with_mne

from collections import defaultdict
import re

from apps.AY2latent_bci.eeg_data import (
    EEGProcessor,
    BCIDatasetArgs,
    create_dataloader_v2,
    chop_and_reshape_signals, # for debug
    invert_reshape_signals,
)

from lingua.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    init_signal_handler,
    get_device_mesh,
    get_is_master,
    setup_env,
    setup_torch_distributed,
    check_model_value_range,
)
from lingua.logger import init_logger
from lingua.metrics import (
    GPUMemoryMonitor,
    LoggingArgs,
    get_num_params,
)

from apps.AY2latent_bci.transformer import (
    DecoderTransformerArgs,
    EncoderDecoder,
)

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger()

@dataclass
class TrainArgs:
    name: str = "lingua"
    dump_dir: str = ""

    seed: int = 42
    grad_acc_steps: int = 1

    gc_collect_freq: int = 1000
    probe_freq: Optional[int] = None

    # Nb optimizer steps to take
    steps: int = 1000

    data: BCIDatasetArgs = field(default_factory=BCIDatasetArgs)
    model: DecoderTransformerArgs = field(default_factory=DecoderTransformerArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)

    # If set to None, eval is run locally otherwise it launches a new job with the given number of gpus
    async_eval_gpus: Optional[int] = None
    eval: Optional[Any] = None

    load_distillation_model: bool = False
    channel_loss_weighting: bool = False
    distill_into_encoder: bool = False
    repa_into_encoder: bool = False
    repa_into_decoder: bool = False

    decoder_loss_weight: float = 1.0
    decoder_repa_weight: float = 1.0
    encoder_mmd_weight: float = 1.0
    encoder_repa_weight: float = 1.0
    encoder_distill_weight: float = 1.0

# @torch.compile()
def process_batch_data(batch, data_processor, loss_weights,):
    with torch.no_grad():
        batch = data_processor.process(**batch)

        return batch, loss_weights

preemption_flag = dict(flag=False)

def set_preemption_flag(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Preemption ! checkpointing asap and exiting.")
    preemption_flag["flag"] = True

def plot_compare_eeg_signal(data,
                            reconst,  
                            eeg_signal=None,  # added this argument to see original signal with no dropout  
                            mne_reconstruction = None,
                            fs=256,
                            batch=0, 
                            sample=0,
                            idx=0,
                            fname_tag="",
                            dir_base="figures"):
    """
    Plot EEG time trace (data & reconst), each channel on a different subplot.
    """
    assert data.shape == reconst.shape

    data = data.T
    reconst = reconst.T
    if eeg_signal is not None:
        eeg_signal = eeg_signal.T
    if mne_reconstruction is not None:
        mne_reconstruction = mne_reconstruction.T

    num_t, chans = data.shape
    t = np.arange(num_t) #/ fs
    print(f"\teeg: {chans=}, {num_t=}")

    best_div = get_best_divisors(chans, max_pad=10)
    dimx, dimy = best_div
    fig, axes = plt.subplots(dimx, dimy, figsize=(24, 12))

    pct_dropout = (np.abs(data).sum(axis=0)==0).sum()/chans
    where_dropout = np.abs(data).sum(axis=0)==0

    if dimx==dimy==1:
        # Single-channel case: (copy-pasted-edited from multi-chan below).
        ch=0
        axes.plot(t, data[:, ch], "b-", linewidth=0.5, alpha=0.4)
        axes.plot(t, reconst[:, ch], "r-", linewidth=0.5, alpha=0.4)
        if eeg_signal is not None:
            axes.plot(t, eeg_signal[:, ch], "g-", linewidth=0.5, alpha=0.4)
        if mne_reconstruction is not None:
            axes.plot(t, mne_reconstruction[:, ch], linestyle="-", color="magenta", linewidth=0.5, alpha=0.4)
        axes.set_xlim(t[0],t[-1])
        axes.tick_params(axis='x', labelsize=10)
        axes.tick_params(axis='y', labelsize=10)
        axes.grid(True)
        axes.text(.98, .98, f"Ch{ch+1}", transform=axes.transAxes, ha='right', va='top', fontsize=12, color='black')
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("Amp")

    else:
        # Multi-channel case: Loop through each subplot and plot something
        ch=-1
        for i in range(dimx):
            for j in range(dimy):
                try:
                    ch+=1
                    # Plot time-domain EEG (offset by channel index)
                    axes[i, j].plot(t, data[:, ch], "b-", linewidth=0.5, alpha=0.4)
                    axes[i, j].plot(t, reconst[:, ch], "r-", linewidth=0.5, alpha=0.4)
                    if eeg_signal is not None and where_dropout[ch]:
                        axes[i, j].plot(t, eeg_signal[:, ch], "g-", linewidth=0.5, alpha=0.4)
                    if mne_reconstruction is not None and where_dropout[ch]:
                        axes[i, j].plot(t, mne_reconstruction[:, ch], linestyle="-", color="magenta", linewidth=0.5, alpha=0.4)
                    axes[i, j].set_xlim(t[0],t[-1])
                    axes[i, j].tick_params(axis='x', labelsize=10)
                    axes[i, j].tick_params(axis='y', labelsize=10)
                    axes[i, j].grid(True)
                    if where_dropout[ch]:
                        axes[i, j].text(.98, .98, f"Ch{ch+1}", transform=axes[i, j].transAxes, ha='right', va='top', fontsize=12, color='green')
                    else:
                        axes[i, j].text(.98, .98, f"Ch{ch+1}", transform=axes[i, j].transAxes, ha='right', va='top', fontsize=12, color='blue')

                    if i==(dimx-1) and j==0:
                        axes[i, j].set_xlabel("Time (s)")
                        axes[i, j].set_ylabel("Amp")

                except:
                    break # If we run out of channels, just break
        
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # leave some space at top for suptitle
    fig.text(0.05, 0.97, "raw", ha='center', va='center', fontsize=16, fontweight='bold', color='green')
    fig.text(0.08, 0.97, "vs.", ha='center', va='center', fontsize=16, fontweight='bold', color='black')
    fig.text(0.12, 0.97, "data in", ha='center', va='center', fontsize=16, fontweight='bold', color='blue')
    fig.text(0.15, 0.97, "vs.", ha='center', va='center', fontsize=16, fontweight='bold', color='black')
    fig.text(0.19, 0.97, "reconst", ha='center', va='center', fontsize=16, fontweight='bold', color='red')
    fig.text(0.22, 0.97, "vs.", ha='center', va='center', fontsize=16, fontweight='bold', color='black')
    fig.text(0.25, 0.97, "mne", ha='center', va='center', fontsize=16, fontweight='bold', color='magenta')
    plt.suptitle(f"EEG{fname_tag} - ({batch=}, {idx=}, {sample=}) - %dropped={pct_dropout:0.3f}", fontsize=16, fontweight='bold')

    plt.savefig(f"{dir_base}/eeg_signal_compare_B{batch}_S{sample}{fname_tag}.png", dpi=300, bbox_inches='tight')
    plt.close()



def get_divisors(n):
    """
    Finds all divisors of a positive integer n.
    """
    if n <= 0:
        return []
    
    divisors = set()
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)

    divs = sorted(list(divisors))  
    return list(zip(divs, divs[::-1]))


def get_best_divisors(chans, max_pad=0):
    """
    Finds the best divisors of a positive integer chans, allowing for padding up to max_pad.
    The best divisors are those that are closest to each other.
    For subplots
    """
    div_diff_best = 1e6
    for pad in range(max_pad):
        a = get_divisors(chans+pad)
        best_div = a[len(a)//2]
        div_diff = abs(best_div[0]-best_div[1]) + 0.25*pad # penalize for padding
        if div_diff < div_diff_best:
            div_diff_best = div_diff
            winner_best_div = best_div

    return winner_best_div


def parse_filename_num_samples(filename):
    """
    Parse filename to extract expected number of samples.
    Example: ds000001_000000_000002_d00_00003_31_1280.pt -> 3 samples
    """
    try:
        parts = filename.removesuffix('.pt').split('_')
        num_samples = int(parts[4])  # The 5th element (index 4) is num_samples
        return num_samples
    except (IndexError, ValueError):
        logger.warning(f"Could not parse num_samples from filename: {filename}")
        return None


def save_reconstructed_file(filename, file_data, export_dir):
    """
    Save a complete reconstructed file with all its samples.

    Args:
        filename: Original filename (e.g., "ds000001_..._.pt")
        file_data: Dict with 'data_original', 'data_reconstructed', 'channel_positions', 'metadata'
        export_dir: Directory to save the file
    """
    output_path = Path(export_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    #JM save pt - Save reconstructed PT file with same structure as input
    output_dict = {
        'data': file_data['data_reconstructed'],        # List of reconstructed samples
        'data_original': file_data['data_original'],    # List of original samples (for comparison)
        'channel_positions': file_data['channel_positions'],
        'metadata': file_data['metadata']
    }

    # Debug: Show reversibility params in metadata
    # if 'reversibility' in file_data['metadata']:
    #     rev = file_data['metadata']['reversibility']
    #     print(f"[METADATA] Saving with reversibility params: global_std={rev.get('global_std', 0)*1e6:.2f} µV, final_std={rev.get('final_std', 0)*1e6:.2f} µV")
    # else:
    #     print(f"[METADATA] ⚠️  No reversibility params in metadata for {filename}!")

    torch.save(output_dict, output_path)  #JM save pt - Actual save to disk

    # Debug: Show how many epochs are valid vs None
    # total_epochs = len(file_data['data_reconstructed'])
    # none_epochs = sum(1 for x in file_data['data_reconstructed'] if x is None)
    # valid_epochs = total_epochs - none_epochs
    # print(f"[DEBUG] Saved {filename}: {valid_epochs}/{total_epochs} valid ({none_epochs} None, {100*none_epochs/total_epochs:.0f}% filtered)")

    logger.info(f"✓ Saved and freed: {filename} ({len(file_data['data_reconstructed'])} samples)")


def check_and_save_complete_files(results_accumulator, export_dir):
    """
    Check for complete files and save them immediately to free memory.

    Args:
        results_accumulator: Dict tracking results by filename
        export_dir: Directory to save files

    Returns:
        List of filenames that were saved (to be removed from accumulator)
    """
    completed_files = []
    for filename, file_data in results_accumulator.items():
        expected = file_data['expected_samples']
        collected = file_data['collected_samples']

        if collected == expected:
            # File is complete - save it
            save_reconstructed_file(filename, file_data, export_dir)
            completed_files.append(filename)

    return completed_files



def unwrap_all_the_signals(model_output, batch, args):
    """
    Unwrap the signals from the model output, latent data, and latent recon.

    This function is used to unwrap the signals from the model output, latent data, and latent recon.

    Inputs:
    - model_output: [B, seqlen, latent_dim]
    - batch: dict -> batch.keys() = ['encoder_input', 'decoder_input', 'target', 't', \
                                    'eeg_signal', 'chan_pos', 'chan_pos_discrete', \
                                    'chan_id', 'seq_lens', 't_coarse']
    - args: argparse.Namespace - args passed in from config file.

    Outputs:
    - model_signal_input_unwrapped: list of numpy arrays, each of shape [num_chans, tc, tf]
    - model_signal_output_unwrapped: list of numpy arrays, each of shape [num_chans, tc, tf]
    - model_position_input_unwrapped: list of numpy arrays, each of shape [num_chans, tc, 3]
    - model_position_discrete_input_unwrapped: list of numpy arrays, each of shape [num_chans, tc, 3]
    - model_position_output_unwrapped: list of numpy arrays, each of shape [num_chans, tc, 3]
    - eeg_signal_unwrapped: list of numpy arrays, each of shape [num_chans, tc, tf]
    - channel_id_unwrapped: list of numpy arrays, each of shape [num_chans, tc]
    - t_coarse_unwrapped: list of numpy arrays, each of shape [num_chans, tc]
    """

    model_input = batch['encoder_input'] #.cpu().numpy()        # Includes channel dropout
    eeg_signal = batch['eeg_signal'] #.cpu().numpy()            # Original eeg signal without channel dropout

    print(f"{batch['seq_lens']=}")
    print(f"{batch['seq_lens'].sum().item()=}")

    if batch['t_coarse'] is not None:
        print(f"{batch['t_coarse'].shape=}")

    print(f"{model_input.shape=}")
    print(f"{model_output.shape=}")

    model_signal_input_unwrapped = []
    model_signal_output_unwrapped = []
    model_position_input_unwrapped = []
    model_position_discrete_input_unwrapped = []
    model_position_output_unwrapped = []
    eeg_signal_unwrapped = [] # without dropout.
    channel_id_unwrapped = []
    t_coarse_unwrapped = []

    seq_lens = batch['seq_lens'].cpu().numpy() 
    seqlen_accum=0

    tf = args.data.num_fine_time_pts
    tc = args.data.seq_len // tf

    # Loop through each sample in batch and unwrap the variable-length sequences
    for i,seqlen in enumerate(seq_lens):
        num_chans = seqlen//tc 
        
        print(f"Sample {i} has seqlen {seqlen} and {num_chans} chans")

        if args.data.cat_chan_xyz_and_eeg:
            mod_in_pos = model_input[seqlen_accum:seqlen_accum+seqlen, :3] # {x,y,z} position channels
            mod_in_sig = model_input[seqlen_accum:seqlen_accum+seqlen, 3:] # tf eeg-signals with channel dropout
            eeg_sig = eeg_signal[seqlen_accum:seqlen_accum+seqlen, 3:] # tf eeg-signals without channel dropout
            mod_out_pos = model_output.squeeze(0)[seqlen_accum:seqlen_accum+seqlen, :3] # {x,y,z} position channels
            mod_out_sig = model_output.squeeze(0)[seqlen_accum:seqlen_accum+seqlen, 3:] # tf eeg-signals
        else:
            mod_in_pos = batch['chan_pos'][seqlen_accum:seqlen_accum+seqlen, :] # {x,y,z} position channels
            mod_in_sig = model_input[seqlen_accum:seqlen_accum+seqlen, :]       # tf eeg-signals with channel dropout
            eeg_sig = eeg_signal[seqlen_accum:seqlen_accum+seqlen, :]       # tf eeg-signals without channel dropout
            mod_out_pos = torch.zeros_like(mod_in_pos)                      # {x,y,z} position channels - not modeled, so just put zeros here.
            mod_out_sig = model_output.squeeze(0)[seqlen_accum:seqlen_accum+seqlen, :] # tf eeg-signals

        t_coarse = batch['t_coarse'][seqlen_accum:seqlen_accum+seqlen, :] if batch['t_coarse'] is not None else None
        chan_id = batch['chan_id'][seqlen_accum:seqlen_accum+seqlen, :] if batch['chan_id'] is not None else None
        mod_in_pos_disc = batch['chan_pos_discrete'][seqlen_accum:seqlen_accum+seqlen, :] # discretized {x,y,z} position channels

        print(f"{seqlen_accum} : {seqlen_accum+seqlen}")

        
        if args.data.use_coarse_time in {"A", "B", "C", "D"}:
            # unwrap (original and reconstructed) signals and positions - inverting chop_and_reshape_signals
            mod_in_sig_unwrapt, mod_in_pos_unwrapt, mod_in_pos_disc_unwrapt, chan_id_unwrapt, tc_unwrapt = invert_reshape_signals(
                                                                                            sig_reshaped=mod_in_sig, 
                                                                                            pos_reshaped=mod_in_pos, 
                                                                                            pos_discrete_reshaped=mod_in_pos_disc, 
                                                                                            id_reshaped=chan_id,
                                                                                            tc_reshaped=t_coarse,
                                                                                            num_chans=num_chans, 
                                                                                            tf=tf,
                                                                                            use_coarse_time=args.data.use_coarse_time,
            )
            mod_out_sig_unwrapt, mod_out_pos_unwrapt, _, _, _ = invert_reshape_signals(
                                                            sig_reshaped=mod_out_sig, 
                                                            pos_reshaped=mod_out_pos, 
                                                            num_chans=num_chans, 
                                                            tf=tf,
                                                            use_coarse_time=args.data.use_coarse_time,
            )
            eeg_sig_unwrapt, _, _, _, _ = invert_reshape_signals(
                                                sig_reshaped=eeg_sig,
                                                num_chans=num_chans, 
                                                tf=tf,
                                                use_coarse_time=args.data.use_coarse_time,
            )
        else:
            print(f"Dont understand {args.data.use_coarse_time=}")

        model_signal_input_unwrapped.append(mod_in_sig_unwrapt.cpu().numpy())
        model_signal_output_unwrapped.append(mod_out_sig_unwrapt.cpu().numpy())
        model_position_input_unwrapped.append(mod_in_pos_unwrapt.cpu().numpy())
        model_position_discrete_input_unwrapped.append(mod_in_pos_disc_unwrapt.cpu().numpy())
        model_position_output_unwrapped.append(mod_out_pos_unwrapt.cpu().numpy())
        eeg_signal_unwrapped.append(eeg_sig_unwrapt.cpu().numpy())
        channel_id_unwrapped.append(chan_id_unwrapt.cpu().numpy())
        try:
            t_coarse_unwrapped.append(tc_unwrapt.cpu().numpy())
        except:
            t_coarse_unwrapped.append(tc_unwrapt) # tc_unwrapt is NoneType probably
        
        seqlen_accum += seqlen


        
        # Some Sanity Check plots to verify that the unwrapping and reshaping are working correctly.
        # These plots should match plots generated in EEGDataset_v2.__iter__, made with same flag.
        check_reshape_plots = False # Plot signals before and after reshaping to verify its working.
        if check_reshape_plots:
            # 1. Plot reshaped signals (input to model)
            if i==0: # save plot only for 1st sample in batch - to match indx0 insider EEGDataset_v2.__iter__
                print(f"Saving plots...")
                for j in range(num_chans):
                    signal = mod_in_sig_unwrapt[j,:].cpu().numpy() 
                    # signal2 = mod_out_sig_unwrapt[j,:].cpu().numpy() 
                    #
                    fig, ax = plt.subplots(1, 1, figsize=(20, 4))
                    ax.plot(signal,color='blue', alpha=0.5)         # plot original data
                    # ax.plot(signal2,color='green', alpha=0.5)     # plot reconstruction
                    ax.scatter(tf*np.arange(tc), signal[::tf], color='red')
                    plt.savefig(f"figures/inspect_reshape_and_invert/test0_ch{j}_final.png", dpi=300, bbox_inches='tight')
                    plt.close()
            # 2. Assert that the unwrapping and reshaping of channel positions worked correctly: shape = [num_chans, tc, 3]
            chan_pos = mod_in_pos_unwrapt.reshape(-1,tc,3)
            for k in range(num_chans):
                tc0 = chan_pos[k,0,:]
                for j in range(1, tc):
                    assert (tc0 == chan_pos[k,j,:]).all().item(), f"chan_pos unwrapping not right for sample {k}, time {j}."
            # 3. Assert that the unwrapping and reshaping for channel id worked correctly: shape = [num_chans, tc]
            for k in range(num_chans):
                assert (chan_id_unwrapt[k]==k).all().item(), f"chan_id unwrapping {k} not right."
            # 4. Assert that the unwrapping and reshaping for coarse_time worked correctly: shape = [num_chan, tc]
            if tc_unwrapt is not None:
                tc0 = tc_unwrapt[0]
                for j in range(1, num_chans):
                    assert (tc0 == tc_unwrapt[j]).all().item(), f"coarse time unwrapping {j} not right."


    return model_signal_input_unwrapped, \
            model_signal_output_unwrapped, \
            model_position_input_unwrapped, \
            model_position_discrete_input_unwrapped, \
            model_position_output_unwrapped, \
            eeg_signal_unwrapped, \
            channel_id_unwrapped, \
            t_coarse_unwrapped



def plot_unwrapped_signals(model_signal_input_unwrapped, 
                            model_signal_output_unwrapped, 
                            eeg_signal_unwrapped,
                            fs,
                            batch_cntr,
                            batch_idx,
                            dir_base,  
                            fname_suptag,
                            plot_eeg_signal_samples,
                            mne_interpolated_signals=None):

        """
        Plot original and EEG reconstructed signals.
        """

        for samp in range(len(model_signal_input_unwrapped)):
            # print(f"sample {samp}")  # Disabled verbose output

            # (1). Plot EEG time course for data and reconstruction on same axis (one ax per channel). One figure per sample.
            if plot_eeg_signal_samples:
                # 1a. Plot with non-dropout signal too.
                plot_compare_eeg_signal(data=model_signal_input_unwrapped[samp],
                                        reconst=model_signal_output_unwrapped[samp],
                                        eeg_signal=eeg_signal_unwrapped[samp],
                                        # mne_reconstruction = mne_interpolated_signals[samp] if mne_interpolated_signals else None, # UNCOMMENT TO PLOT MNE INTERPOLATED SIGNALS
                                        fs=fs,
                                        batch=batch_cntr,
                                        sample=samp,
                                        idx=batch_idx[samp].item(),
                                        fname_tag=""+fname_suptag,
                                        dir_base=dir_base,
                )

#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#


def evaluate(args: TrainArgs):

    plot_eeg_signal_samples = False      # Plot raw eeg for data and model reconstruction for single samples
    compute_mne_interpolated_signals = False

    sample_steps = 50    # for diffusion process in .sample - Default is 50
    cfg = 1.0            # for diffusion process in .sample - Default is 1.0 (i.e., no cfg)

    num_batches = 5
    batch_cntr = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    tmp_sample_idx = []
    tmp_filenames = []

    dir_base = f'figures/zuna/cfg{cfg}/'
    print(f"Saving output figures to: {dir_base=}")
    os.makedirs(dir_base, exist_ok=True)

    # saving pt files - setup export directory and results accumulator
    export_dir = args.data.export_dir
    print(f"Will save reconstructed pt files to: {export_dir}")
    os.makedirs(export_dir, exist_ok=True)
    results_accumulator = {} # tracks samples by filename until file is complete

    fs = args.data.sample_rate
    num_t = args.data.seq_len


    with ExitStack() as context_stack:
        if get_is_master():
            os.makedirs(args.dump_dir, exist_ok=True)
            dump_config(args, Path(args.dump_dir) / "config.yaml")
        init_logger(Path(args.dump_dir) / "train.log")
        init_signal_handler(set_preemption_flag)  # For handling preemption signals.
        setup_env(args.env)

        setup_torch_distributed(args.distributed, device=device)
        world_mesh = get_device_mesh(args.distributed, device=device)
        logger.info(f"Starting job: {args.name}")

        # build dataloader - need dp world size and rank
        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
        if args.distributed.dp_shard > 1:
            dp_rank = dp_rank * world_mesh["dp_shard"].size() + world_mesh["dp_shard"].get_local_rank()
            dp_degree *= world_mesh["dp_shard"].size()

        logger.info(f"Running on dp rank : {dp_rank}")
        logger.info(f"Running on dp size : {dp_degree}")

        torch.manual_seed(args.seed)
        logger.info("Building model")

        
        # In your shell, set your HF_TOKEN environment variable:
        # export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"
        def load_model_args_local(config_path: str) -> DecoderTransformerArgs:
            cfg = OmegaConf.load(config_path)
            cfg_obj = OmegaConf.to_container(cfg, resolve=True)
            return dataclass_from_dict(DecoderTransformerArgs, cfg_obj.get("model", {}))

        def load_model_args_from_hf(repo_id: str, config_filename: str = "config.json") -> DecoderTransformerArgs:
            config_path = hf_hub_download(repo_id=repo_id, filename=config_filename, token=True)
            with open(config_path, "r") as f:
                cfg = json.load(f)
            return dataclass_from_dict(DecoderTransformerArgs, cfg["model"])

        REPO_ID = "Zyphra/ZUNA"
        WEIGHTS = "model-00001-of-00001.safetensors"
        # arg_path = "src/zuna/inference/AY2l/lingua/apps/AY2latent_bci/configs/config_bci_eval.yaml"
        CONFIG  = "config.json"

        # EITHER FROM LOCAL OR FROM HF FOR CONFIGS
        # model_args = load_model_args_local(arg_path)
        model_args = load_model_args_from_hf(REPO_ID, CONFIG)

        weights_path = hf_hub_download(repo_id=REPO_ID, filename=WEIGHTS, token=True)
        sd_st_raw = safe_load(weights_path, device="cpu")

        # Normalize: strip leading "model." if present
        sd_st = {k.removeprefix("model."): v for k, v in sd_st_raw.items()}

        model = EncoderDecoder(model_args).to(device)
        sd_st_on_dev = {k: v.to(device) for k, v in sd_st.items()}
        model.load_state_dict(sd_st_on_dev, strict=True)
        model.eval()

        logger.info("Model is built !")
        model_param_count = get_num_params(model)

        if device.type == "cuda":
            model.sample = torch.compile(model.sample)
            model.encoder = torch.compile(model.encoder)

        model.eval()

        check_model_value_range(model, range=10.0, std=1.0)

        # log model size
        logger.info(f"Model size: {model_param_count:,} total parameters")

        if device.type == "cuda":
            gpu_memory_monitor = GPUMemoryMonitor("cuda")
            logger.info(
                f"GPU capacity: {gpu_memory_monitor.device_name} ({gpu_memory_monitor.device_index}) "
                f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
            )
            logger.info(f"GPU memory usage: {gpu_memory_monitor}")
        else:
            logger.info(f"Running on CPU")

        gc.disable()

        # Make seed unique per GPU/rank by adding rank to base seed
        rank_seed = args.seed + dp_rank
        torch.manual_seed(rank_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(rank_seed)

        logger.info(f"Setting torch seed to {rank_seed} for rank {dp_rank}")
        
        # Also make numpy and random seeds unique per rank
        np.random.seed(rank_seed)
        random.seed(rank_seed)

        # Create dataloader
        print("Entering create dataloader on rank", dp_rank)
        data_loader = create_dataloader_v2(args.data, args.seed, dp_rank)
        print("Finishing create dataloader on rank", dp_rank)


        epoch = 0 # if using nonlocal epoch
        def make_batch_iterator(dataloader, data_args):  # Use with IterableDataset.
            """
            Moving sequence packing into Dataset/Dataloader/Collator. Too slow when done here.
            """
            nonlocal epoch
            print("Creating batch iterator of dataloader with length", len(dataloader), "and dataset of length", len(dataloader.dataset))

            eeg_sig_norm = data_args.data_norm
            eeg_sig_clip = data_args.data_clip

            while True:
                epoch += 1
                logger.info(f"Starting epoch: {epoch}")
                for idx,batch in enumerate(dataloader):

                    eeg_signal = batch['eeg_signal']
                    eeg_signal = eeg_signal/eeg_sig_norm # Divide by eeg_sig_norm to normalize the data and change its STD.

                    if eeg_sig_clip is not None:
                        print(f"Clipping input at +/-{eeg_sig_clip}")
                        eeg_signal = eeg_signal.clamp(min=-eeg_sig_clip, max=eeg_sig_clip)

                    yield {
                        'eeg_signal': eeg_signal, # pass out the clipped and normalized eeg signal.
                        'chan_pos': batch['chan_pos'],
                        'chan_pos_discrete': batch['chan_pos_discrete'],
                        'chan_id': batch['chan_id'],
                        't_coarse': batch['t_coarse'],
                        'chan_dropout': batch['chan_dropout'],
                        'seq_lens': batch['seq_lens'],
                        'idx': batch['ids'],
                        'dataset_id': batch['dataset_id'],
                        'filename': batch['filename'],           # Pass through filename
                        'sample_idx': batch['sample_idx'],       # Pass through sample_idx
                        'metadata': batch['metadata'],           # Pass through metadata
                    }

                print("Finished epoch", epoch)

        batch_iterator = make_batch_iterator(data_loader, args.data)
        print("Entering create batch iterator on rank", dp_rank)

        for p in model.parameters():
            p.requires_grad = False #(False for eval, True for training)

        data_processor = EEGProcessor(args.data).to(device)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # Debug - Track (filename, sample_idx) occurrences as a matrix
        sample_occurrence_matrix = defaultdict(lambda: defaultdict(int))  # [filename][sample_idx] = count
        file_max_samples = {}  # Track expected max samples per file

        while True:
            batch = next(batch_iterator)
            batch_cntr += 1

            # Break after epoch 1 completes to avoid duplicate processing during inference
            if epoch > 1:
                print(f"\n[DEBUG] Stopping at batch {batch_cntr}: epoch {epoch} > 1, breaking to avoid duplicate processing")
                break

            eeg_signal = batch['eeg_signal']
            batch_idx = batch.pop('idx', None)
            batch_dataset_id = batch.pop('dataset_id', None)   # NOTE: pop takes them out of batch. if left in, breaks things below and not training on these.
            batch_filenames = batch.pop('filename', None)         
            batch_sample_indices = batch.pop('sample_idx', None)    
            batch_metadata_list = batch.pop('metadata', None)      

            # Debug - Populate occurrence matrix for this batch
            if batch_filenames and batch_sample_indices:
                for filename, sample_idx in zip(batch_filenames, batch_sample_indices):
                    sample_occurrence_matrix[filename][sample_idx] += 1

                    # Track max samples expected per file (from metadata if available)
                    if filename not in file_max_samples:
                        # Try to get from metadata or infer from filename
                        
                        match = re.search(r'_d\d+_(\d+)_', filename)  # Extract num samples from filename like d30_00064_
                        if match:
                            file_max_samples[filename] = int(match.group(1))
                        else:
                            file_max_samples[filename] = 64  # Default assumption

            #Debug - Print matrix every 50 batches to show duplicates/missing
            if batch_cntr % 50 == 0:
                print(f"\n{'='*80}")
                print(f"[DEBUG MATRIX] After {batch_cntr} batches:")
                print(f"{'='*80}")
                for filename in sorted(sample_occurrence_matrix.keys()):
                    max_samples = file_max_samples.get(filename, 64)
                    counts = sample_occurrence_matrix[filename]

                    # Count issues
                    zeros = sum(1 for i in range(max_samples) if counts[i] == 0)
                    ones = sum(1 for i in range(max_samples) if counts[i] == 1)
                    duplicates = sum(1 for i in range(max_samples) if counts[i] > 1)
                    total_occurrences = sum(counts.values())

                    print(f"\n{filename} (expected {max_samples} samples):")
                    print(f"  0x (missing): {zeros}, 1x (good): {ones}, 2+x (duplicates): {duplicates}")
                    print(f"  Total occurrences: {total_occurrences}")

                    if duplicates > 0:
                        # Show which indices are duplicated
                        dup_indices = [i for i in range(max_samples) if counts[i] > 1]
                        print(f"  ⚠️  DUPLICATES at indices: {dup_indices[:20]}")
                        print(f"      Counts: {[counts[i] for i in dup_indices[:20]]}")

                    if zeros > 0 and zeros < max_samples:  # Don't show if ALL are zero
                        # Show which indices are missing
                        missing_indices = [i for i in range(max_samples) if counts[i] == 0]
                        print(f"  ⚠️  MISSING indices: {missing_indices[:20]}")
                print(f"{'='*80}\n")


            with torch.no_grad():
                batch = data_processor.process(**batch)

            batch = {k: v.to(device, non_blocking=(device.type=="cuda")) for  k, v in batch.items()}

            tf = args.data.num_fine_time_pts
            tc = args.data.seq_len // tf

            if args.data.use_coarse_time=="C":
                tc = 1 # HARDCODE: USE THIS when chop_signals_only, using first tf seconds in signal.

            # ## Options for tok_idx.  Choose 1 in config.
            if args.model.tok_idx_type is None:
                tok_idx = None          # this will just use args.model.max_seqlen to construct 1D-RoPE (but requires max_seqlen way too long).
            elif args.model.tok_idx_type == "t_coarse" and args.model.rope_dim==1:
                tok_idx = batch['t_coarse'].cpu().unsqueeze(0)   # this ignores channel and just uses coarse time in 1D-RoPE
            elif args.model.tok_idx_type == "chan_id" and args.model.rope_dim==1:
                tok_idx = batch['chan_id'].cpu().unsqueeze(0)       # this uses channel id in 1D-RoPE  # this is same as hstack(arange(seq_lens)) below when seq_len = num_chans, ie chop_signals_only
            elif args.model.tok_idx_type == "stack_arange_seqlen" and args.model.rope_dim==1:
                tok_idx = torch.hstack(
                    [torch.arange(sl) for sl in list(batch['seq_lens'].cpu().numpy())]
                ).unsqueeze(0).unsqueeze(-1)                                                # This has a different tok_id value for each element in sequence (chan or tc).
            elif args.model.tok_idx_type == "{x,y,z,tc}" and args.model.rope_dim==4: 
                chan_pos_discrete = batch['chan_pos_discrete'].cpu().unsqueeze(0)      # [1, seqlen, 3]
                t_coarse = batch['t_coarse'].cpu().unsqueeze(0)         # [1, seqlen, 1]
                tok_idx = torch.cat((chan_pos_discrete,t_coarse), dim=2)
            else:
                print(f"Dont understand {args.model.tok_idx_type=} and {args.model.rope_dim}")
                die 


            with torch.no_grad():
                z, inference_at_step = model.sample(
                    encoder_input=batch['encoder_input'].unsqueeze(0),
                    seq_lens=batch['seq_lens'],
                    tok_idx=tok_idx,
                    cfg=cfg,
                    sample_steps=sample_steps,
                )    

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     

            signals_to_plot = []
            # signals_to_plot = inference_at_step # UNCOMMENT IF YOU WANT TO PLOT THE INTERMEDIATE STEPS OF THE DIFFUSION PROCESS
                                                  # NOTE: If computing reconstruction-based metrics, we need only the final sample from the diffusion process.

            signals_to_plot.append(z) # Always append the final sample from the diffusion process

            for step in range(len(signals_to_plot)):

                print(f"Processing step {step} of {len(signals_to_plot)}")
                z = signals_to_plot[step]
                fname_suptag="_step"+str(step)
                if step == len(signals_to_plot) - 1:
                    fname_suptag = "_stepFinal"

                # Unwrap signals
                model_signal_input_unwrapped, \
                model_signal_output_unwrapped, \
                model_position_input_unwrapped, \
                model_position_discrete_input_unwrapped, \
                model_position_output_unwrapped, \
                eeg_signal_unwrapped, \
                channel_id_unwrapped, \
                t_coarse_unwrapped = unwrap_all_the_signals(model_output=z, 
                                                            batch=batch, 
                                                            args=args)    

                # eeg_signal_unwrapped: original data | is list, each item has ch*timepoints
                # model_signal_input_unwrapped: input with dropped channels
                # model_signal_output_unwrapped, output of the model
                # chan_pos = model_position_input_unwrapped[0].reshape(-1,tc,3)[:,0,:] #channel position requires this reshape

                # Save pt files - accumulate results for each sample
                eeg_sig_norm = args.data.data_norm # IMPORTANT: Reverse normalization (was divided by 10.0 in make_batch_iterator)

                # #Debug: Show which samples are in this batch
                # if batch_cntr % 10 == 0:  # Print every 10th batch to avoid spam
                #     print(f"[DEBUG] Batch {batch_cntr}: Processing {len(model_signal_output_unwrapped)} samples")
                #     print(f"  Files: {set(batch_filenames)}")
                #     print(f"  Sample indices: {batch_sample_indices[:10]}{'...' if len(batch_sample_indices) > 10 else ''}")

                for i in range(len(model_signal_output_unwrapped)):
                    filename = batch_filenames[i]
                    sample_idx = batch_sample_indices[i]
                    metadata = batch_metadata_list[i]
                    tmp_sample_idx.append(sample_idx)
                    tmp_filenames.append(filename)

                    # Initialize file entry if first time seeing this file
                    if filename not in results_accumulator:
                        num_samples = parse_filename_num_samples(filename)
                        if num_samples is None:
                            logger.warning(f"Skipping file with unparseable filename: {filename}")
                            continue

                        results_accumulator[filename] = {
                            'expected_samples': num_samples,
                            'collected_samples': 0,
                            'data_original': [None] * num_samples,
                            'data_reconstructed': [None] * num_samples,
                            'channel_positions': [None] * num_samples,
                            'metadata': metadata
                        }

                    # Store this sample's results (multiply by eeg_sig_norm to reverse normalization)
                    file_entry = results_accumulator[filename]
                    # Debug: Track which sample indices are being processed
                    if file_entry['collected_samples'] == 0:  # First sample from this file
                        print(f"[DEBUG] Starting file {filename}: expecting {file_entry['expected_samples']} samples")
                    if file_entry['collected_samples'] < 5 or file_entry['collected_samples'] >= file_entry['expected_samples'] - 3:
                        print(f"  Storing sample_idx={sample_idx} (#{file_entry['collected_samples']+1}/{file_entry['expected_samples']})")

                    file_entry['data_original'][sample_idx] = eeg_signal_unwrapped[i] * eeg_sig_norm
                    file_entry['data_reconstructed'][sample_idx] = model_signal_output_unwrapped[i] * eeg_sig_norm
                    file_entry['channel_positions'][sample_idx] = model_position_input_unwrapped[i].reshape(-1, tc, 3)[:, 0, :]
                    file_entry['collected_samples'] += 1


                # Check if any files are complete and save them
                completed = check_and_save_complete_files(results_accumulator, export_dir)
                for filename in completed:
                    del results_accumulator[filename]  # Free memory

                # Apply MNE interpolation to dropped-out channels
                if compute_mne_interpolated_signals:
                    chan_pos_list = [model_position_input_unwrapped[i].reshape(-1, tc, 3)[:, 0, :] for i in range(len(model_signal_input_unwrapped))]
                    #
                    mne_interpolated_signals = interpolate_signals_with_mne(
                        signals=model_signal_input_unwrapped,
                        channel_positions=chan_pos_list,
                        sampling_rate=fs,
                        mark_zero_variance=True
                    )
                else:
                    mne_interpolated_signals = None

                # Plot signals
                plot_unwrapped_signals(model_signal_input_unwrapped, 
                                        model_signal_output_unwrapped, 
                                        eeg_signal_unwrapped, 
                                        fs,
                                        batch_cntr,
                                        batch_idx,
                                        dir_base,
                                        fname_suptag,  
                                        plot_eeg_signal_samples,
                                        mne_interpolated_signals=mne_interpolated_signals)


            # # Here if you want to only do a certain number of batches (like for making a couple plots))
            # if batch_cntr >= num_batches:
            #     break

            # # Here if you want to only do a certain number of epochs - like go thru whole dataset once (as when computng eval metric stats)
            if epoch > 1:
                break

        #saving pt files - save any remaining incomplete files at the end
        if results_accumulator:
            logger.info(f"\nProcessing complete. Saving {len(results_accumulator)} remaining files...")
            for filename, file_data in results_accumulator.items():
                expected = file_data['expected_samples']
                collected = file_data['collected_samples']

                if collected == expected:
                    # Complete file that hasn't been saved yet
                    save_reconstructed_file(filename, file_data, export_dir)
                else:
                    # Incomplete file - save with warning
                    # logger.warning(f"Incomplete file: {filename} ({collected}/{expected} samples collected)")  # Disabled verbose output
                    # You can choose to save incomplete files or skip them
                    # For now, let's save them with a flag
                    file_data['metadata']['incomplete'] = True
                    file_data['metadata']['collected_samples'] = collected
                    file_data['metadata']['expected_samples'] = expected
                    save_reconstructed_file(filename, file_data, export_dir)

            logger.info(f"All files saved to: {export_dir}")

    # Debug - Analyze tmp_sample_idx and tmp_filenames before final check
    # Disabled verbose output - uncomment for debugging
    # print("\n" + "="*80)
    # print("DEBUG: Analyzing processed samples (tmp_sample_idx, tmp_filenames)")
    # print("="*80)
    #
    # from collections import Counter
    # counts_idx = Counter(tmp_sample_idx)
    # counts_filenames = Counter(tmp_filenames)
    #
    # print(f"\nTotal samples processed: {len(tmp_sample_idx)}")
    # print(f"Unique filenames: {len(set(tmp_filenames))}")
    #
    # print("\nFilename occurrence counts:")
    # for filename, count in sorted(counts_filenames.items()):
    #     print(f"  {filename}: {count} times")
    #
    # print("\nChecking for duplicate sample indices:")
    # duplicates = {idx: count for idx, count in counts_idx.items() if count > 1}
    # if duplicates:
    #     print(f"  Found {len(duplicates)} duplicate indices!")
    #     for idx, count in sorted(duplicates.items())[:10]:
    #         print(f"    Index {idx}: appeared {count} times")
    # else:
    #     print("  No duplicates found")
    #
    # # Check per-file
    # print("\nPer-file analysis:")
    # for filename in sorted(set(tmp_filenames)):
    #     indices = [idx for idx, f in zip(tmp_sample_idx, tmp_filenames) if f == filename]
    #     expected = 64  # Assuming 64 samples per file
    #     print(f"\n  {filename}:")
    #     print(f"    Processed: {len(indices)} times")
    #     print(f"    Unique indices: {len(set(indices))}")
    #     print(f"    Expected: {expected}")
    #
    #     # Check for missing indices
    #     unique_indices = set(indices)
    #     missing = [i for i in range(expected) if i not in unique_indices]
    #     if missing:
    #         print(f"    Missing indices: {missing[:20]}{'...' if len(missing) > 20 else ''}")
    #
    #     # Check for duplicates within this file
    #     dup_count = len(indices) - len(unique_indices)
    #     if dup_count > 0:
    #         print(f"    ⚠️  {dup_count} duplicate entries!")
    #
    # print("="*80 + "\n")

    # import pdb; pdb.set_trace()

    # Debug - Final verification: Check all saved PT files for None/missing samples
    print("\n" + "="*80)
    print("FINAL VERIFICATION: Checking all saved PT files")
    print("="*80)

    export_path = Path(export_dir)
    saved_pt_files = sorted(export_path.glob("*.pt"))

    print(f"\nFound {len(saved_pt_files)} saved PT files\n")

    total_files = 0
    total_samples = 0
    total_none = 0
    total_valid = 0

    for pt_file in saved_pt_files:
        try:
            data = torch.load(pt_file, weights_only=False)
            reconstructed = data.get('data', [])

            n_samples = len(reconstructed)
            n_none = sum(1 for x in reconstructed if x is None)
            n_valid = n_samples - n_none

            total_files += 1
            total_samples += n_samples
            total_none += n_none
            total_valid += n_valid

            status = "✓" if n_none == 0 else "⚠️"
            print(f"{status} {pt_file.name}")
            print(f"   Total: {n_samples} | Valid: {n_valid} | None: {n_none} ({100*n_none/n_samples if n_samples > 0 else 0:.0f}%)")

            if n_none > 0:
                # Show which indices are None
                none_indices = [i for i, x in enumerate(reconstructed) if x is None]
                print(f"   None at indices: {none_indices[:30]}{'...' if len(none_indices) > 30 else ''}")

        except Exception as e:
            print(f"✗ {pt_file.name}: ERROR - {e}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total files: {total_files}")
    print(f"Total samples: {total_samples}")
    print(f"Valid samples: {total_valid} ({100*total_valid/total_samples if total_samples > 0 else 0:.1f}%)")
    print(f"None samples: {total_none} ({100*total_none/total_samples if total_samples > 0 else 0:.1f}%)")
    print("="*80 + "\n")

#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#


def main():
    """
    """
    cli_args = OmegaConf.from_cli()

    file_cfig = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfig = OmegaConf.structured(TrainArgs())
    cfig = OmegaConf.merge(default_cfig, file_cfig, cli_args)
    cfig = OmegaConf.to_object(cfig)

    evaluate(cfig)


if __name__ == "__main__":
    main()
