
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


# 1st, setup tmux and docker with lingua.sh
#   >> "pip install zuna" or something?

# 2nd, run something like:
#   >> CUDA_VISIBLE_DEVICES=1 python3 src/zuna/inference/AY2l/lingua/apps/AY2latent_bci/eeg_eval.py config=src/zuna/inference/AY2l/lingua/apps/AY2latent_bci/configs/config_bci_eval.yaml


from copy import deepcopy
import gc
import logging
import os
# import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional
import random
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.distributed.checkpoint.stateful import Stateful
# from torch.distributed._tensor import DTensor
import matplotlib.pyplot as plt

# To load model from HuggingFace.
import json
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load


from lingua.args import dataclass_from_dict, dump_config, flatten_dict
from lingua.checkpoint import CheckpointArgs, CheckpointManager, load_from_checkpoint

from utils_pt_mne import interpolate_signals_with_mne

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
    get_world_size,
    setup_env,
    setup_torch_distributed,
    check_model_value_range,
)
from lingua.logger import init_logger
from lingua.metrics import (
    GPUMemoryMonitor,
    LoggingArgs,
    MetricLogger,
    get_num_params,
)
from lingua.optim import OptimArgs, build_optimizer
from apps.AY2latent_bci.transformer import (
    DecoderTransformerArgs,
    EncoderDecoder,
)
from lingua.probe import AutoProbeD

from dotenv import load_dotenv
load_dotenv() # Load WANDB_API_KEY from .env file

logger = logging.getLogger()

# SAVE_RECONSTRUCTION_PTS = True  # Flag to save reconstructions and latents into pt files so we can run classifier on them


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
    optim: OptimArgs = field(default_factory=OptimArgs)
    model: DecoderTransformerArgs = field(default_factory=DecoderTransformerArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)
    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
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


@dataclass
class TrainState(Stateful):
    step: int  # Nb of steps taken by the optimizer
    acc_step: int  # Nb of accumulation steps done since last optimizer step
    scheduler: lr_scheduler.LambdaLR

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "acc_step": self.acc_step,
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]
        self.scheduler.load_state_dict(state_dict["scheduler"])

def validate_train_args(args: TrainArgs,):
    assert args.dump_dir, "Dump dir not set"

    if args.checkpoint.path is None:
        logger.info(f"Setting checkpoint path to {str(Path(args.dump_dir) / 'checkpoints')}")
        args.checkpoint.path = str(Path(args.dump_dir) / "checkpoints")

    # if using local filesystem, check if data_dir exists.
    if not args.data.use_b2:
        assert os.path.exists(args.data.data_dir), f"{args.data.data_dir} doesn't exist"

    if (
        args.distributed.dp_replicate
        * args.distributed.dp_shard
        * args.distributed.tp_size
        != get_world_size()
    ):
        assert get_world_size() % args.distributed.dp_shard == 0
        args.distributed.dp_replicate = get_world_size() // args.distributed.dp_shard

        assert args.distributed.dp_replicate % args.distributed.tp_size == 0
        args.distributed.dp_replicate = (
            args.distributed.dp_replicate // args.distributed.tp_size
        )

        logger.warning(
            f"Setting Data Parallel size to {args.distributed.dp_replicate * args.distributed.dp_shard}"
        )
        assert (
            args.distributed.dp_replicate
            * args.distributed.dp_shard
            * args.distributed.tp_size
            == get_world_size()
        )

        if args.distributed.fsdp_type == "no_shard":
            assert (
                args.distributed.dp_shard == 1
                and args.distributed.dp_replicate == get_world_size()
            )

    if args.distributed.tp_size == 1:
        logger.warning(
            "Tensor parallelism has not been tested for a while, use at your own risk"
        )

    if args.logging.wandb is not None:
        args.logging.wandb.name = args.name

    if args.probe_freq is not None:
        assert (
            args.distributed.tp_size == 1
        ), "Probing not supported with tensor parallelism"
        assert (
            args.distributed.selective_activation_checkpointing is False
        ), "Probing not supported with selective activation checkpointing"

preemption_flag = dict(flag=False)

def set_preemption_flag(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Preemption ! checkpointing asap and exiting.")
    preemption_flag["flag"] = True




def plot_compare_eeg_signal(data,
                            reconst,  
                            eeg_signal=None,  # (CW) - added this argument to see original signal with no dropout  
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
            print(f"sample {samp}")

            # (1). Plot EEG time course for data and reconstruction on same axis (one ax per channel). One figure per sample.
            if plot_eeg_signal_samples:
                # 1a. Plot with non-dropout signal too.
                plot_compare_eeg_signal(data=model_signal_input_unwrapped[samp],
                                        reconst=model_signal_output_unwrapped[samp],
                                        eeg_signal=eeg_signal_unwrapped[samp],
                                        mne_reconstruction = mne_interpolated_signals[samp] if mne_interpolated_signals else None, # UNCOMMENT TO PLOT MNE INTERPOLATED SIGNALS
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

    plot_eeg_signal_samples = True      # Plot raw eeg for data and model reconstruction for single samples
    compute_mne_interpolated_signals = True

    sample_steps = 50    # for diffusion process in .sample - Default is 50
    cfg = 1.0            # for diffusion process in .sample - Default is 1.0 (i.e., no cfg)

    num_batches = 5
    batch_cntr = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu") 

    dir_base = 'figures/' + '/'.join(args.checkpoint.init_ckpt_path.split('/')[-3:]) + f'/cfg{cfg}'
    print(f"Saving output figures to: {dir_base=}")
    os.makedirs(dir_base, exist_ok=True)

    fs = args.data.sample_rate
    num_t = args.data.seq_len


    with ExitStack() as context_stack:
        validate_train_args(
            args,
        )

        if get_is_master():
            os.makedirs(args.dump_dir, exist_ok=True)
            dump_config(args, Path(args.dump_dir) / "config.yaml")
        init_logger(Path(args.dump_dir) / "train.log")
        init_signal_handler(set_preemption_flag)  # For handling preemption signals.
        setup_env(args.env)

        setup_torch_distributed(args.distributed, device=device)
        world_mesh = get_device_mesh(args.distributed, device=device)
        logger.info(f"Starting job: {args.name}")

        # build dataloader
        # need dp world size and rank
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

        if False:
            # Load the model from the checkpoint.
            with torch.device("meta"):
                model = EncoderDecoder(args.model) # load from yaml file

            logger.info("Model is built !")
            model_param_count = get_num_params(model)

            model.sample = torch.compile(model.sample)
            model.encoder = torch.compile(model.encoder)
            model = model.to_empty(device=device) 

            if device.type == "cuda":
                if args.checkpoint.init_ckpt_path:
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        torch.manual_seed(args.model.seed)
                        model.init_weights()
                    check_model_value_range(model, range=10.0, std=1.0)
                    logger.info(f"!!!! Loading initial model from {args.checkpoint.init_ckpt_path} !!!! \n\n")
                    load_from_checkpoint(args.checkpoint.init_ckpt_path, model, model_key="model") # Put model_key="" if its directly the model checkpoint
                    logger.info("!!!!!!!!!!! Model loaded from checkpoint completed !!!!!!!!!!!")
                    check_model_value_range(model, range=10.0, std=1.0)
                else:
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        torch.manual_seed(args.model.seed)
                        model.init_weights()
            check_model_value_range(model, range=10.0, std=1.0)

            # log model size
            logger.info(f"Model size: {model_param_count:,} total parameters")


        if True:
            print("LOAD THE MODEL FROM HUGGINGFACE.")
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
                # expects {"model": {...}} like you showed
                return dataclass_from_dict(DecoderTransformerArgs, cfg["model"])

            REPO_ID = "Zyphra/ZUNA"
            WEIGHTS = "model-00001-of-00001.safetensors"
            arg_path = "src/zuna/inference/AY2l/lingua/apps/AY2latent_bci/configs/config_bci_eval.yaml"
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

            model.sample = torch.compile(model.sample)
            model.encoder = torch.compile(model.encoder)

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


        ## DONT THINK WE NEED THIS FOR EVALS
        # build optimizer after apply parallelisms to the model
        optimizer, scheduler = build_optimizer(model, args.optim, args.steps,)
        # data_loader_state = init_dataloader_state_from_args(
        #     args.data, dp_rank, dp_degree
        # )
        
        train_state = TrainState(
            step=0,
            acc_step=0,
            # data_loader_state=data_loader_state,
            scheduler=scheduler,
        )
        
        checkpoint = CheckpointManager.instantiate_and_make_dir(args.checkpoint)
        checkpoint.load(model, optimizer, train_state, world_mesh)
        # Either load from latest checkpoint or start from scratch
        if args.probe_freq is not None:
            if get_is_master():
                os.makedirs(Path(args.dump_dir) / "probe", exist_ok=True)
            torch.distributed.barrier()
            probe = AutoProbeD(
                model,
                (
                    Path(args.dump_dir) / "probe" / f"probe.{dp_rank}.jsonl"
                    if (dp_rank % 128 == 0)
                    else None
                ),
            )

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

        model.eval()
        metric_logger = context_stack.enter_context(
            MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args)
        )

        print("Entering create dataloader on rank", dp_rank)
        data_loader = create_dataloader_v2(args.data, args.seed, dp_rank)
        print("Finishing create dataloader on rank", dp_rank)


        epoch = 0 # if using nonlocal epoch
        def make_batch_iterator(dataloader):  # (CW) Use with IterableDataset.
            """
            Moving sequence packing into Dataset/Dataloader/Collator. Too slow when done here.
            """
            nonlocal epoch
            # epoch = 0 # if not using nonlocal epoch
            # dataloader.sampler.set_epoch(epoch)
            print("Creating batch iterator of dataloader with length", len(dataloader), "and dataset of length", len(dataloader.dataset))

            eeg_sig_norm = 10.0 # normalization factor for eeg signal.
            eeg_sig_clip = 1.0 #None  # clipping factor for eeg signal.

            while True:
                epoch += 1
                logger.info(f"Starting epoch: {epoch}")
                for idx,batch in enumerate(dataloader):

                    eeg_signal = batch['eeg_signal']

                    eeg_signal = eeg_signal/eeg_sig_norm # Divide by eeg_sig_norm to normalize the data and change its STD.

                    if eeg_sig_clip is not None:
                        print(f"Clipping input at +/-{eeg_sig_clip}")
                        eeg_signal = eeg_signal.clamp(min=-eeg_sig_clip, max=eeg_sig_clip) # 

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
                    }

                # dataloader.sampler.set_epoch(epoch)
                print("Finished epoch", epoch)

        batch_iterator = make_batch_iterator(data_loader)
        print("Entering create batch iterator on rank", dp_rank)

        torch_profiler = None
        #make sure all model parameters require gradients
        for p in model.parameters():
            p.requires_grad = False #(False for eval, True for training)

        data_processor = EEGProcessor(args.data).to(device)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
        while True:
            batch = next(batch_iterator)     
            batch_cntr += 1

            eeg_signal = batch['eeg_signal']
            # batch_ids = batch.pop('ids', None)
            batch_idx = batch.pop('idx', None)
            batch_dataset_id = batch.pop('dataset_id', None)   # NOTE: pop takes them out of batch. (CW) - if left in, breaks things below and not training on these.
            with torch.no_grad(): 
                batch = data_processor.process(**batch)                             #  > option 3. (CW)
            
            batch = {k: v.to(device, non_blocking=(device.type=="cuda")) for  k, v in batch.items()}

            tf = args.data.num_fine_time_pts
            tc = args.data.seq_len // tf

            if args.data.use_coarse_time=="C":
                tc = 1 # (CW) - HARDCODE: USE THIS when chop_signals_only, using first tf seconds in signal.

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
            # signals_to_plot = inference_at_step # [ inference_at_step[-2] ] # UNCOMMENT IF YOU WANT TO PLOT THE INTERMEDIATE STEPS OF THE DIFFUSION PROCESS
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

                # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)  
                
                # eeg_signal_unwrapped: original data | is list, each item has ch*timepoints
                # model_signal_input_unwrapped: input with dropped channels 
                # model_signal_output_unwrapped, output of the model 
                # chan_pos = model_position_input_unwrapped[0].reshape(-1,tc,3)[:,0,:] #channel position requires this reshape
                

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
            if batch_cntr >= num_batches:
                break

            # # Here if you want to only do a certain number of epochs (like for computng eval metric stats)
            # if epoch > 1:
            #     break

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
