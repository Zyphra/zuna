
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


# 1st, setup tmux and docker with lingua.sh
#   >> "pip install zuna" or something?

# 2nd, run something like:
#   >> CUDA_VISIBLE_DEVICES=0 python3 src/zuna/inference/AY2l/lingua/apps/AY2latent_bci/eeg_eval.py config=src/zuna/inference/AY2l/lingua/apps/AY2latent_bci/configs/config_bci_eval.yaml


import numpy as np
from scipy.signal import welch, hilbert
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from copy import deepcopy
import gc
import logging
import os
import sys
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
# import xformers.profiler
from torch.optim import lr_scheduler
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed._tensor import DTensor

# from lingua.apps.AY2latent.data_lean import STFTProcessor
from lingua.args import dataclass_from_dict, dump_config, flatten_dict
from lingua.checkpoint import CheckpointArgs, CheckpointManager, load_from_checkpoint

from utils_pt_mne import interpolate_signals_with_mne #, egi_montage_subsampling

from apps.AY2latent_bci.eeg_data import (
    # EEGDataset_v2, 
    EEGProcessor, 
    BCIDatasetArgs, 
    # create_dataloader,
    create_dataloader_v2,
    chop_and_reshape_signals, # for debug
    invert_reshape_signals,
)

from lingua.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    init_signal_handler,
    dist_mean_dict,
    get_device_mesh,
    get_is_master,
    get_world_size,
    parallelize_model,
    setup_env,
    setup_torch_distributed,
    clean_env,
    requeue_slurm_job,
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
# from lingua.profiling import ProfilerArgs #, maybe_run_profiler (CW)
# from lingua.tokenizer import build_tokenizer
from apps.AY2latent_bci.transformer import (
    DecoderTransformerArgs,
    EncoderDecoder,
    get_num_flop_per_token,
    build_fsdp_grouping_plan,
    # tp_parallelize,
    get_no_recompute_ops,
)
from lingua.probe import AutoProbeD
from lingua.stool import StoolArgs, launch_job

import wandb
from dotenv import load_dotenv
load_dotenv() # Load WANDB_API_KEY from .env file

from torch._dynamo.decorators import mark_static_address
import functools
logger = logging.getLogger()

LOAD_THE_MODEL = True           # Flag to load model onto GPU or not. If False, just explore data.
# SAVE_RECONSTRUCTION_PTS = True  # Flag to save reconstructions and latents into pt files so we can run classifier on them



def compute_mae(y_true, y_pred):
    """
    Compute Mean Absolute Error between two signals.
    """
    # Ensure inputs are numpy arrays for vectorization
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Calculate the absolute difference, then take the mean
    mae = np.mean(np.abs(y_true - y_pred))
    
    return mae

def compute_nmse(y_true, y_pred):
    """
    Compute Normalized Mean Square Error between two signals.
    """
    mse = np.mean((y_true - y_pred)**2)
    normalization = np.mean(y_true**2)
    return mse / normalization # maybe 10 * np.log10(mse / normalization) for dB?


def compute_snr(y_true, y_pred):
    """
    Compute Signal-to-Noise Ratio between two signals.
    """
    # Power of the clean signal
    sig_power = np.sum(y_true**2)
    
    # Power of the noise (the difference)
    noise_power = np.sum((y_true - y_pred)**2)
    
    # Compute ratio in dB
    snr = 10 * np.log10(sig_power / noise_power)
    return snr

def compute_pcc(y_true, y_pred):
    """
    Compute Pearson Correlation Coefficient between two signals.
    """
    # This returns a 2x2 matrix; the [0, 1] element is the r value
    return np.corrcoef(y_true, y_pred)[0, 1]


@dataclass
class TrainArgs:
    name: str = "lingua"
    dump_dir: str = ""

    seed: int = 42

    # Number of gradient accumulation steps
    # Total batch size is batch_size*grad_acc_steps
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
    # profiling: ProfilerArgs = field(default_factory=ProfilerArgs) (CW)
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

    # print(f"Dawgg, process_batch_data")
    # import IPython; print('\n\n Debug:'); IPython.embed(); import time;  time.sleep(0.3)

    with torch.no_grad():
        batch = data_processor.process(**batch)

        return batch, loss_weights


@dataclass
class TrainState(Stateful):
    step: int  # Nb of steps taken by the optimizer
    acc_step: int  # Nb of accumulation steps done since last optimizer step
    scheduler: lr_scheduler.LambdaLR
    # data_loader_state: PackTokensState

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "acc_step": self.acc_step,
            # "data_loader_state": self.data_loader_state,
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]
        # self.data_loader_state = PackTokensState(**state_dict["data_loader_state"])
        self.scheduler.load_state_dict(state_dict["scheduler"])

def validate_train_args(args: TrainArgs,):
    assert args.dump_dir, "Dump dir not set"

    if args.checkpoint.path is None:
        logger.info(f"Setting checkpoint path to {str(Path(args.dump_dir) / 'checkpoints')}")
        args.checkpoint.path = str(Path(args.dump_dir) / "checkpoints")

    # (CW) - if using local filesystem, check if data_dir exists.
    if not args.data.use_b2:
        assert os.path.exists(args.data.data_dir), f"{args.data.data_dir} doesn't exist" # (CW) - replaced with this (NEWEST)
    #
    # assert os.path.exists(args.data.data_path), f"{args.data.data_path} doesn't exist" # (CW) - was this first
    #
    # for data_path in args.data.data_paths:
    #     assert os.path.exists(data_path), f"{data_path} doesn't exist" # (CW) - replaced with this second (OLD)

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

    # args.model.max_seqlen = 4096 # (CW) int(args.data.sample_duration_seconds * args.data.sample_rate)
    #                              # this needs to be bigger than 1280*1.5 to include registers.

    if args.distributed.tp_size == 1:
        logger.warning(
            "Tensor parallelism has not been tested for a while, use at your own risk"
        )

    ## (CW)
    # assert (
    #     args.probe_freq != args.profiling.mem_steps
    # ), "Don't profile during probe step"
    # assert (
    #     args.probe_freq != args.profiling.profile_steps
    # ), "Don't profile during probe step"

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


def every_n_steps(train_state, freq, acc_step=None, acc_freq=None):
    test = train_state.step % freq == 0
    if acc_step is not None:
        test = test and (train_state.acc_step == acc_step)
    elif acc_freq is not None:
        test = test and ((train_state.acc_step % acc_freq) == 0)
    return test

def reshape_eeg_signal(eeg_signal, polyphase_factor=1): # (CW)
    """
    eeg_signal.shape = [B,C,T] = [B, 64, 1280] 
    EncoderDecoder model expects [B,T,C].
    Here, we permute dims 2 & 1 with x.permute(0, 2, 1)
    Can also implement Polyphase here to downsample (T,seqlen)
    and put it into (C,dim) -> [B, 256, 320] 
    """

    # print("Inside reshape_eeg_signal")
    # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)

    eeg_signal = eeg_signal.permute(0,2,1)
    if polyphase_factor > 1:
        print("Need to implement polyphase.")
    return eeg_signal



def plot_compare_eeg_signal(data,
                            reconst,  
                            mse_value,
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

    num_t, chans = data.shape
    t = np.arange(num_t) #/ fs
    print(f"\teeg: {chans=}, {num_t=}")

    # dim = int(np.ceil(np.sqrt(chans)))
    best_div = get_best_divisors(chans, max_pad=10)
    dimx, dimy = best_div
    fig, axes = plt.subplots(dimx, dimy, figsize=(24, 12))

    pct_dropout = (np.abs(data).sum(axis=0)==0).sum()/chans
    where_dropout = np.abs(data).sum(axis=0)==0


    # print(f"In plot_compare_eeg_signal, ... ")
    # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

    if eeg_signal is not None:
        MSE_dropout = np.abs(reconst[:,where_dropout] - eeg_signal[:,where_dropout]).mean()
        MSE_nondrop = np.abs(reconst[:,~where_dropout] - eeg_signal[:,~where_dropout]).mean()
    if mne_reconstruction is not None:
        #jm - Transpose MNE reconstruction from (channels, times) to (times, channels)
        mne_reconstruction = mne_reconstruction.T
        MSE_mne_dropout = np.abs(reconst[:,where_dropout] - mne_reconstruction[:,where_dropout]).mean()


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
                    # axes[i, j].set_ylim(-0.3,0.3) # hardcoded. Comment out.
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
    plt.suptitle(f"EEG{fname_tag} - ({batch=}, {idx=}, {sample=}) - MSE={mse_value:0.5f} - %dropped={pct_dropout:0.3f}", fontsize=16, fontweight='bold')

    if eeg_signal is not None:
        fig.text(0.8, 0.97, f"MSE_do={MSE_dropout:0.3f}", ha='center', va='center', fontsize=16, fontweight='bold', color='green')
        fig.text(0.9, 0.97, f"MSE_~do={MSE_nondrop:0.3f}", ha='center', va='center', fontsize=16, fontweight='bold', color='blue')
    if mne_reconstruction is not None:
        fig.text(0.8, 0.95, f"MSE_mne={MSE_mne_dropout:0.3f}", ha='center', va='center', fontsize=16, fontweight='bold', color='magenta')


    # (CW) - try to use dark background for plots.
    if True:
        plt.style.use('dark_background')

    plt.savefig(f"{dir_base}/eeg_signal_compare_B{batch}_S{sample}{fname_tag}.png", dpi=300, bbox_inches='tight')
    plt.close()




def plot_compare_eeg_position(data,
                              reconst,  
                              mse_value,
                              batch=0, 
                              sample=0,
                              idx=0,
                              fname_tag="",
                              dir_base="figures",
):
    """
    Plot EEG electrode position (data & reconst) over coarse time scale.
    """
    assert data.shape == reconst.shape

    tc = data.shape[1]//3 # coarse time

    # Split out coarse_time from {x,z,y} channels.
    data = data.reshape(-1,tc,3)
    reconst = reconst.reshape(-1,tc,3)

    # Sanity check: channel positions in data should be same for all time points
    #       Not true for reconst - but is desirable.
    xxx = data[:,0,:]
    for i in range(1,tc):
        yyy = data[:,i,:]
        assert (yyy == xxx).all()

    if tc ==  1:
        dimx, dimy = 1,1
    elif tc == 10:
        dimx, dimy = 2,5
    elif tc == 40:
        dimx, dimy = 5,8
    else:
        print(f"In plot_compare_eeg_position, tc is unexpected: {dimx=}, {dimy=}, {tc=}")
        import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

    assert dimx*dimy == tc

    tf_sec = 5.0/tc # fine-time window in seconds for each coarse time chunk (full signal is 5 seconds for now)

    fig, axes = plt.subplots(dimx, dimy, figsize=(24, 12), subplot_kw=dict(projection="3d"))

    # Get max/min for each axis, across data & reconst for subplot consistency
    max_x = max( data[:,:,0].max(), reconst[:,:,0].max() )
    min_x = min( data[:,:,0].min(), reconst[:,:,0].min() )
    max_y = max( data[:,:,1].max(), reconst[:,:,1].max() )
    min_y = min( data[:,:,1].min(), reconst[:,:,1].min() )
    max_z = max( data[:,:,2].max(), reconst[:,:,2].max() )
    min_z = min( data[:,:,2].min(), reconst[:,:,2].min() )



    if tc == 1:
        tc = tc-1 # just to make the indexing work.
        # If only one coarse time point (chop_signals_only)
        axes.view_init(elev=20, azim=120)
        axes.set_box_aspect([1, 1, 1])
        axes.set_xlim(min_x, max_x)
        axes.set_ylim(min_y, max_y)
        axes.set_zlim(min_z, max_z)
        #
        xd = data[:, tc, 0]
        yd = data[:, tc, 1]
        zd = data[:, tc, 2]
        axes.scatter(xd, yd, zd, marker='o', s=20, facecolors='none', edgecolors='b', alpha=0.3)
        #
        xr = reconst[:, tc, 0]
        yr = reconst[:, tc, 1]
        zr = reconst[:, tc, 2]
        axes.scatter(xr, yr, zr, marker='o', s=15, facecolors='r', edgecolors='r', alpha=0.3)
        #
        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_zlabel('z')
        axes.set_title(f"tc = {tc*tf_sec}-{(tc+1)*tf_sec} secs ")

    else:
        # Loop through each subplot and plot channel positions at that coarse time.
        tc=-1
        for i in range(dimx):
            for j in range(dimy):
                tc+=1
                axes[i, j].view_init(elev=20, azim=120)
                axes[i, j].set_box_aspect([1, 1, 1])
                axes[i, j].set_xlim(min_x, max_x)
                axes[i, j].set_ylim(min_y, max_y)
                axes[i, j].set_zlim(min_z, max_z)
                #
                xd = data[:, tc, 0]
                yd = data[:, tc, 1]
                zd = data[:, tc, 2]
                axes[i, j].scatter(xd, yd, zd, marker='o', s=20, facecolors='none', edgecolors='b', alpha=0.3)
                #
                xr = reconst[:, tc, 0]
                yr = reconst[:, tc, 1]
                zr = reconst[:, tc, 2]
                axes[i, j].scatter(xr, yr, zr, marker='o', s=15, facecolors='r', edgecolors='r', alpha=0.3)

                if tc==0:
                    axes[i, j].set_xlabel('x')
                    axes[i, j].set_ylabel('y')
                    axes[i, j].set_zlabel('z')
                axes[i, j].set_title(f"tc = {tc*tf_sec}-{(tc+1)*tf_sec} secs ")
        
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # leave some space at top for suptitle
    fig.text(0.27, 0.97, "data", ha='center', va='center', fontsize=16, fontweight='bold', color='blue')
    fig.text(0.30, 0.97, "vs.", ha='center', va='center', fontsize=16, fontweight='bold', color='black')
    fig.text(0.34, 0.97, "reconst", ha='center', va='center', fontsize=16, fontweight='bold', color='red')
    plt.suptitle(f"EEG - ({batch=}, {idx=}, {sample=}) - MSE={mse_value:0.5f}", fontsize=16, fontweight='bold')

    plt.savefig(f"{dir_base}/eeg_position_compare_B{batch}_S{sample}{fname_tag}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_compare_fft(data, 
                     reconst,
                     mse_value,
                     mse_value_do,
                     mse_value_nodo,
                     freqs, 
                     batch=0, 
                     sample=0,
                     idx=0,
                     fname_tag="",
                     dir_base="figures"):
    
    """
    Plot FFT spectrum (data & reconst), each channel on a different subplot.
    """

    assert data.shape == reconst.shape

    data = data.T
    reconst = reconst.T

    num_f, chans = data.shape
    print(f"\tfft: {chans=}, {num_f=}")

    # dim = int(np.ceil(np.sqrt(chans)))
    best_div = get_best_divisors(chans, max_pad=10)
    dimx, dimy = best_div
    fig, axes = plt.subplots(dimx, dimy, figsize=(24, 12))

    if dimx==dimy==1:

        # Single channel case: (copy-pasted-edited from multi-chan case below)
        ch=0
        # Plot FFT of EEG
        axes.plot(freqs, data[:, ch], "b-", linewidth=0.5, alpha=0.4)
        axes.plot(freqs, reconst[:, ch], "r-", linewidth=0.5, alpha=0.4)
        axes.set_xlim(freqs[0],freqs[-1])
        axes.tick_params(axis='x', labelsize=10)
        axes.tick_params(axis='y', labelsize=10)
        axes.grid(True)
        axes.text(.98, .98, f"Ch{ch+1}", transform=axes.transAxes, ha='right', va='top', fontsize=12, color='black')
        axes.set_xlabel("Freq (hz)")
        axes.set_ylabel("Amp")

    else:
        # Multi-channel case:
        # Loop through each subplot and plot something
        ch=-1
        for i in range(dimx):
            for j in range(dimy):
                try:  
                    ch+=1
                    # Plot FFT of EEG
                    axes[i, j].plot(freqs, data[:, ch], "b-", linewidth=0.5, alpha=0.4)
                    axes[i, j].plot(freqs, reconst[:, ch], "r-", linewidth=0.5, alpha=0.4)
                    axes[i, j].set_xlim(freqs[0],freqs[-1])
                    axes[i, j].tick_params(axis='x', labelsize=10)
                    axes[i, j].tick_params(axis='y', labelsize=10)
                    axes[i, j].grid(True)
                    axes[i, j].text(.98, .98, f"Ch{ch+1}", transform=axes[i, j].transAxes, ha='right', va='top', fontsize=12, color='black')
                    
                    if i==(dimx-1) and j==0:
                        axes[i, j].set_xlabel("Freq (hz)")
                        axes[i, j].set_ylabel("Amp")
            
                except:
                    break # If we run out of channels, just break
        
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # leave some space at top for suptitle
    fig.text(0.05, 0.97, "data", ha='center', va='center', fontsize=16, fontweight='bold', color='blue')
    fig.text(0.08, 0.97, "vs.", ha='center', va='center', fontsize=16, fontweight='bold', color='black')
    fig.text(0.12, 0.97, "reconst", ha='center', va='center', fontsize=16, fontweight='bold', color='red')
    plt.suptitle(f"EEG FFT - ({batch=}, {idx=}, {sample=}) - MSE={mse_value:0.5f}, MSE_do={mse_value_do:0.5f}, MSE_nodo={mse_value_nodo:0.5f}", fontsize=16, fontweight='bold')

    plt.savefig(f"{dir_base}/fft_compare_B{batch}_S{sample}{fname_tag}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_compare_latents(data,
                         reconst,  
                         mse_value,
                         batch=0, 
                         sample=0,
                         idx=0,
                         fname_tag="",
                         dir_base="figures"):

    """
    Plot latents from encoder operating on (data & reconst), each channel on a different subplot.
    """
    assert data.shape == reconst.shape

    # print(f"Inside plot_compare_latents:")
    # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)
    
    data = data.T
    reconst = reconst.T

    num_t, chans = data.shape
    t = np.arange(num_t) #/ fs
    print(f"\tlat: {chans=}, {num_t=}")

    # dim = int(np.ceil(np.sqrt(chans)))
    best_div = get_best_divisors(chans, max_pad=10)
    dimx, dimy = best_div
    fig, axes = plt.subplots(dimx, dimy, figsize=(24, 12))

    if dimx==dimy==1:
        # Single chan case
        ch=0
        # Plot time-domain EEG (offset by channel index)
        axes.plot(t, data[:, ch], "b-", linewidth=0.5, alpha=0.4)
        axes.plot(t, reconst[:, ch], "r-", linewidth=0.5, alpha=0.4)
        axes.set_xlim(t[0],t[-1])
        axes.tick_params(axis='x', labelsize=10)
        axes.tick_params(axis='y', labelsize=10)
        axes.grid(True)
        axes.text(.98, .98, f"dim {ch+1}", transform=axes.transAxes, ha='right', va='top', fontsize=12, color='black')
        axes.set_xlabel("Latent Sequence")
        axes.set_ylabel("Amp")

    else:
        # Multi-chan case
        # Loop through each subplot and plot something
        ch=-1
        for i in range(dimx):
            for j in range(dimy):
                try:
                    ch+=1
                    # Plot time-domain EEG (offset by channel index)
                    axes[i, j].plot(t, data[:, ch], "b-", linewidth=0.5, alpha=0.4)
                    axes[i, j].plot(t, reconst[:, ch], "r-", linewidth=0.5, alpha=0.4)
                    axes[i, j].set_xlim(t[0],t[-1])
                    axes[i, j].tick_params(axis='x', labelsize=10)
                    axes[i, j].tick_params(axis='y', labelsize=10)
                    axes[i, j].grid(True)
                    axes[i, j].text(.98, .98, f"dim {ch+1}", transform=axes[i, j].transAxes, ha='right', va='top', fontsize=12, color='black')
                    if i==(dimx-1) and j==0:
                        axes[i, j].set_xlabel("Latent Sequence")
                        axes[i, j].set_ylabel("Amp")

                except:
                    break # If we run out of channels, just break
        
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # leave some space at top for suptitle
    fig.text(0.13, 0.97, "data", ha='center', va='center', fontsize=16, fontweight='bold', color='blue')
    fig.text(0.16, 0.97, "vs.", ha='center', va='center', fontsize=16, fontweight='bold', color='black')
    fig.text(0.20, 0.97, "reconst", ha='center', va='center', fontsize=16, fontweight='bold', color='red')
    plt.suptitle(f"Encoder Latents - ({batch=}, {idx=}, {sample=}) - MSE={mse_value:0.5f}", fontsize=16, fontweight='bold')

    plt.savefig(f"{dir_base}/latents_compare_B{batch}_S{sample}{fname_tag}.png", dpi=300, bbox_inches='tight')
    plt.close()

    
    # print("Inside plot_compare_latents")
    # print(f"{reconst.sum(axis=0)=}")
    # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)


def get_distinct_colors(n):
    # Use multiple good colormaps and stack them
    base_maps = [
        plt.cm.tab20,
        plt.cm.tab20b,
        plt.cm.tab20c,
        plt.cm.Set1,
        plt.cm.Set2,
        plt.cm.Set3,
        plt.cm.Pastel1,
        plt.cm.Pastel2,
    ]
    
    colors = []
    for cmap in base_maps:
        colors.extend([to_hex(cmap(i)) for i in np.linspace(0, 1, cmap.N)])
        if len(colors) >= n:
            break

    return colors[:n]


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
    """
    div_diff_best = 1e6
    for pad in range(max_pad):
        a = get_divisors(chans+pad)
        best_div = a[len(a)//2]
        div_diff = abs(best_div[0]-best_div[1]) + 0.25*pad # penalize for padding
        if div_diff < div_diff_best:
            div_diff_best = div_diff
            winner_best_div = best_div

        # print(f"With chans = {chans}+{pad}: {best_div=}, {div_diff=}")

    return winner_best_div



def unwrap_all_the_signals(model_output, latent_data, latent_recon, batch, args):
    """
    Unwrap the signals from the model output, latent data, and latent recon.

    This function is used to unwrap the signals from the model output, latent data, and latent recon.

    Inputs:
    - model_output: [B, seqlen, latent_dim]
    - latent_data: [B, seqlen, latent_dim] or None
    - latent_recon: [B, seqlen, latent_dim] or None
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
    - latent_data_unwrapped: list of numpy arrays, each of shape [num_chans, tc, latent_dim]
    - latent_recon_unwrapped: list of numpy arrays, each of shape [num_chans, tc, latent_dim]
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

    if latent_data is not None and latent_recon is not None:
        print(f"{latent_recon.shape=}")
        print(f"{latent_data.shape=}")
        latent_data_unwrapped = []
        latent_recon_unwrapped = []

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
            
        lat_data = latent_data.squeeze(0)[seqlen_accum:seqlen_accum+seqlen, :] if latent_data is not None else None # latent computed from eeg_signals
        lat_recon = latent_recon.squeeze(0)[seqlen_accum:seqlen_accum+seqlen, :] if latent_recon is not None else None # latent recomputed from reconstructed signals

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
            lat_data_unwrapt, _, _, _, _ = invert_reshape_signals(
                                                sig_reshaped=lat_data,
                                                num_chans=num_chans, 
                                                tf=tf+3 if args.data.cat_chan_xyz_and_eeg else tf,
                                                use_coarse_time=args.data.use_coarse_time,
            )
            lat_recon_unwrapt, _, _, _, _ = invert_reshape_signals(
                                                sig_reshaped=lat_recon,
                                                num_chans=num_chans, 
                                                tf=tf+3 if args.data.cat_chan_xyz_and_eeg else tf,
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
        latent_data_unwrapped.append(lat_data_unwrapt.cpu().numpy())
        latent_recon_unwrapped.append(lat_recon_unwrapt.cpu().numpy())
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
                    signal = mod_in_sig_unwrapt[j,:].cpu().numpy()      # model input should match before and after
                    # signal2 = mod_out_sig_unwrapt[j,:].cpu().numpy()    # should be close I think, right?
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
            latent_data_unwrapped, \
            latent_recon_unwrapped, \
            t_coarse_unwrapped



def compute_sig_FFT(signal_unwrapped, fs):
    """
    Compute FFT of a list of signals (each element is a sample).
    """
    fft_signal_unwrapped = []
    for samp in range(len(signal_unwrapped)):
        model_in_sig = signal_unwrapped[samp]
        #
        num_t = model_in_sig.shape[-1]
        freqs = rfftfreq(num_t, 1/fs)
        #
        fft_data = np.abs(rfft(model_in_sig, axis=1))
        data_norms = np.linalg.norm(fft_data, axis=1) 
        fft_data = fft_data / (data_norms[:, np.newaxis] + 1e-6)
        #
        fft_signal_unwrapped.append(fft_data)
    
    return fft_signal_unwrapped, freqs


def compute_reconstruction_metrics_unwrapped_signals(model_signal_input_unwrapped, 
                                                    model_signal_output_unwrapped,  
                                                    eeg_signal_unwrapped, 
                                                    model_position_input_unwrapped=None, 
                                                    model_position_output_unwrapped=None, 
                                                    latent_data_unwrapped=None, 
                                                    latent_recon_unwrapped=None,
                                                    fft_signal_input_unwrapped=None,
                                                    fft_signal_output_unwrapped=None):
    """
    Compute reconstruction metrics (MAE, NMSE, SNR, PCC) between latents, EEG signals, and FFTs.
    """

    # 1. Compute MSE between latent_data and latent_recon 
    if latent_data_unwrapped is not None and latent_recon_unwrapped is not None:
        MSE_samp_latent = []
        MAE_samp_latent = []
        NMSE_samp_latent = []
        SNR_samp_latent = []
        PCC_samp_latent = []
        for samp in range(len(latent_data_unwrapped)):
            latent_data_sample = latent_data_unwrapped[samp]
            latent_recon_sample = latent_recon_unwrapped[samp]
            MSE = np.abs(latent_data_sample - latent_recon_sample).mean() 
            MAE = compute_mae(latent_data_sample, latent_recon_sample)
            NMSE = compute_nmse(latent_data_sample, latent_recon_sample)
            SNR = compute_snr(latent_data_sample, latent_recon_sample)
            PCC = compute_pcc(latent_data_sample, latent_recon_sample)
            MSE_samp_latent.append(MSE)
            MAE_samp_latent.append(MAE)
            NMSE_samp_latent.append(NMSE)
            SNR_samp_latent.append(SNR)
            PCC_samp_latent.append(PCC)
    else:
        MSE_samp_latent = [None]
        MAE_samp_latent = [None]
        NMSE_samp_latent = [None]
        SNR_samp_latent = [None]
        PCC_samp_latent = [None]


    # 2. Compute MSE between raw data and reconstruction for EEG (across each sample individually).
    #    Do it separately for dropped-out (do) and non-dropped-out (nodo) channels.
    MSE_samp_EEG_pos = []
    MSE_samp_EEG_sig = []
    MSE_samp_EEG_sig_do = []
    MSE_samp_EEG_sig_nodo = []
    #
    MAE_samp_EEG_pos = []
    NMSE_samp_EEG_pos = []
    SNR_samp_EEG_pos = []
    PCC_samp_EEG_pos = []
    #
    MAE_samp_EEG_sig = []
    NMSE_samp_EEG_sig = []
    SNR_samp_EEG_sig = []
    PCC_samp_EEG_sig = []
    #
    MAE_samp_EEG_sig_do = []
    NMSE_samp_EEG_sig_do = []
    SNR_samp_EEG_sig_do = []
    PCC_samp_EEG_sig_do = []
    #
    MAE_samp_EEG_sig_nodo = []
    NMSE_samp_EEG_sig_nodo = []
    SNR_samp_EEG_sig_nodo = []
    PCC_samp_EEG_sig_nodo = []
    #
    for samp in range(len(model_signal_input_unwrapped)):
        dropped_chans = np.abs(model_signal_input_unwrapped[samp]).sum(axis=1)==0

        model_in_sig = eeg_signal_unwrapped[samp] 
        model_out_sig = model_signal_output_unwrapped[samp]

        if model_position_input_unwrapped is not None and model_position_output_unwrapped is not None:
            model_in_pos = model_position_input_unwrapped[samp]
            model_out_pos = model_position_output_unwrapped[samp]
            #
            MSE_EEG_pos = np.abs(model_in_pos - model_out_pos).mean() # mean square error btwn data and reconst
            #
            MAE_EEG_pos = compute_mae(model_in_pos, model_out_pos)
            NMSE_EEG_pos = compute_nmse(model_in_pos, model_out_pos)
            SNR_EEG_pos = compute_snr(model_in_pos, model_out_pos)
            PCC_EEG_pos = compute_pcc(model_in_pos, model_out_pos)
        #
        MSE_EEG_sig = np.abs(model_in_sig - model_out_sig).mean() # mean square error btwn data and reconst
        #
        MAE_EEG_sig = compute_mae(model_in_sig, model_out_sig)
        NMSE_EEG_sig = compute_nmse(model_in_sig, model_out_sig)
        SNR_EEG_sig = compute_snr(model_in_sig, model_out_sig)
        PCC_EEG_sig = compute_pcc(model_in_sig, model_out_sig)
        #
        MSE_EEG_sig_do = np.abs(model_in_sig[dropped_chans] - model_out_sig[dropped_chans]).mean() # mean square error btwn data and reconst on dropped-out chans
        MSE_EEG_sig_nodo = np.abs(model_in_sig[~dropped_chans] - model_out_sig[~dropped_chans]).mean() # mean square error btwn data and reconst on non-dropped chans
        #
        if sum(dropped_chans) > 0:
            MAE_EEG_sig_do = compute_mae(model_in_sig[dropped_chans], model_out_sig[dropped_chans])
            NMSE_EEG_sig_do = compute_nmse(model_in_sig[dropped_chans], model_out_sig[dropped_chans])
            SNR_EEG_sig_do = compute_snr(model_in_sig[dropped_chans], model_out_sig[dropped_chans])
            PCC_EEG_sig_do = compute_pcc(model_in_sig[dropped_chans], model_out_sig[dropped_chans])
        else:
            MAE_EEG_sig_do = np.nan
            NMSE_EEG_sig_do = np.nan
            SNR_EEG_sig_do = np.nan
            PCC_EEG_sig_do = np.nan
        #
        if sum(~dropped_chans) > 0:
            MAE_EEG_sig_nodo = compute_mae(model_in_sig[~dropped_chans], model_out_sig[~dropped_chans])
            NMSE_EEG_sig_nodo = compute_nmse(model_in_sig[~dropped_chans], model_out_sig[~dropped_chans])
            SNR_EEG_sig_nodo = compute_snr(model_in_sig[~dropped_chans], model_out_sig[~dropped_chans])
            PCC_EEG_sig_nodo = compute_pcc(model_in_sig[~dropped_chans], model_out_sig[~dropped_chans])
        else:
            MAE_EEG_sig_nodo = np.nan
            NMSE_EEG_sig_nodo = np.nan
            SNR_EEG_sig_nodo = np.nan
            PCC_EEG_sig_nodo = np.nan
        #
        
        MSE_samp_EEG_sig.append(MSE_EEG_sig)
        #
        MSE_samp_EEG_sig_do.append(MSE_EEG_sig_do)
        MSE_samp_EEG_sig_nodo.append(MSE_EEG_sig_nodo)
        #
        if model_position_input_unwrapped is not None and model_position_output_unwrapped is not None:
            MSE_samp_EEG_pos.append(MSE_EEG_pos)
            #
            MAE_samp_EEG_pos.append(MAE_EEG_pos)
            NMSE_samp_EEG_pos.append(NMSE_EEG_pos)
            SNR_samp_EEG_pos.append(SNR_EEG_pos)
            PCC_samp_EEG_pos.append(PCC_EEG_pos)
        #
        MAE_samp_EEG_sig.append(MAE_EEG_sig)
        NMSE_samp_EEG_sig.append(NMSE_EEG_sig)
        SNR_samp_EEG_sig.append(SNR_EEG_sig)
        PCC_samp_EEG_sig.append(PCC_EEG_sig)
        #
        MAE_samp_EEG_sig_do.append(MAE_EEG_sig_do)
        NMSE_samp_EEG_sig_do.append(NMSE_EEG_sig_do)
        SNR_samp_EEG_sig_do.append(SNR_EEG_sig_do)
        PCC_samp_EEG_sig_do.append(PCC_EEG_sig_do)
        #
        MAE_samp_EEG_sig_nodo.append(MAE_EEG_sig_nodo)
        NMSE_samp_EEG_sig_nodo.append(NMSE_EEG_sig_nodo)
        SNR_samp_EEG_sig_nodo.append(SNR_EEG_sig_nodo)
        PCC_samp_EEG_sig_nodo.append(PCC_EEG_sig_nodo)


    if model_position_input_unwrapped is None and model_position_output_unwrapped is None:
        MSE_samp_EEG_pos = [None]
        #
        MAE_samp_EEG_pos = [None]
        NMSE_samp_EEG_pos = [None]
        SNR_samp_EEG_pos = [None]
        PCC_samp_EEG_sig = [None]
        


    # 3. Compute MSE between raw data and reconstruction for FFT (across each sample individually).
    #    Do it separately for dropped-out (do) and non-dropped-out (nodo) channels.
    if fft_signal_input_unwrapped is not None and fft_signal_output_unwrapped is not None:
        MSE_samp_FFT = []
        MSE_samp_FFT_do = []
        MSE_samp_FFT_nodo = []
        #
        MAE_samp_FFT = []
        NMSE_samp_FFT = []
        SNR_samp_FFT = []
        PCC_samp_FFT = []
        #
        MAE_samp_FFT_do = []
        NMSE_samp_FFT_do = []
        SNR_samp_FFT_do = []
        PCC_samp_FFT_do = []
        #
        MAE_samp_FFT_nodo = []
        NMSE_samp_FFT_nodo = []
        SNR_samp_FFT_nodo = []
        PCC_samp_FFT_nodo = []
        #   
        for samp in range(len(model_signal_input_unwrapped)):
            dropped_chans = np.abs(model_signal_input_unwrapped[samp]).sum(axis=1)==0

            fft_sample_data = fft_signal_input_unwrapped[samp]
            fft_sample_recon = fft_signal_output_unwrapped[samp]
            MSEf = np.abs(fft_sample_data - fft_sample_recon).mean() # mean square error btwn data and reconst FFTs   
            MSE_FFT_do = np.abs(fft_sample_data[dropped_chans] - fft_sample_recon[dropped_chans]).mean() # mean square error btwn data and reconst on dropped-out chans
            MSE_FFT_nodo = np.abs(fft_sample_data[~dropped_chans] - fft_sample_recon[~dropped_chans]).mean() # mean square error btwn data and reconst on non-dropped chans
            #
            MAE_FFT = compute_mae(fft_sample_data, fft_sample_recon)
            NMSE_FFT = compute_nmse(fft_sample_data, fft_sample_recon)
            SNR_FFT = compute_snr(fft_sample_data, fft_sample_recon)
            PCC_FFT = compute_pcc(fft_sample_data, fft_sample_recon)
            #
            if sum(dropped_chans) > 0:
                MAE_FFT_do = compute_mae(fft_sample_data[dropped_chans], fft_sample_recon[dropped_chans])
                NMSE_FFT_do = compute_nmse(fft_sample_data[dropped_chans], fft_sample_recon[dropped_chans])
                SNR_FFT_do = compute_snr(fft_sample_data[dropped_chans], fft_sample_recon[dropped_chans])
                PCC_FFT_do = compute_pcc(fft_sample_data[dropped_chans], fft_sample_recon[dropped_chans])
            else:
                MAE_FFT_do = np.nan
                NMSE_FFT_do = np.nan
                SNR_FFT_do = np.nan
                PCC_FFT_do = np.nan
            #
            if sum(~dropped_chans) > 0:
                MAE_FFT_nodo = compute_mae(fft_sample_data[~dropped_chans], fft_sample_recon[~dropped_chans])
                NMSE_FFT_nodo = compute_nmse(fft_sample_data[~dropped_chans], fft_sample_recon[~dropped_chans])
                SNR_FFT_nodo = compute_snr(fft_sample_data[~dropped_chans], fft_sample_recon[~dropped_chans])
                PCC_FFT_nodo = compute_pcc(fft_sample_data[~dropped_chans], fft_sample_recon[~dropped_chans])
            else:
                MAE_FFT_nodo = np.nan
                NMSE_FFT_nodo = np.nan
                SNR_FFT_nodo = np.nan
                PCC_FFT_nodo = np.nan
            #

            MSE_samp_FFT.append(MSEf)
            MSE_samp_FFT_do.append(MSE_FFT_do)
            MSE_samp_FFT_nodo.append(MSE_FFT_nodo)
            #
            MAE_samp_FFT.append(MAE_FFT)
            NMSE_samp_FFT.append(NMSE_FFT)
            SNR_samp_FFT.append(SNR_FFT)
            PCC_samp_FFT.append(PCC_FFT)
            #
            MAE_samp_FFT_do.append(MAE_FFT_do)
            NMSE_samp_FFT_do.append(NMSE_FFT_do)
            SNR_samp_FFT_do.append(SNR_FFT_do)
            PCC_samp_FFT_do.append(PCC_FFT_do)
            #
            MAE_samp_FFT_nodo.append(MAE_FFT_nodo)
            NMSE_samp_FFT_nodo.append(NMSE_FFT_nodo)
            SNR_samp_FFT_nodo.append(SNR_FFT_nodo)
            PCC_samp_FFT_nodo.append(PCC_FFT_nodo)
    else:
        MSE_samp_FFT = [None]
        MSE_samp_FFT_do = [None]
        MSE_samp_FFT_nodo = [None]
        #
        MAE_samp_FFT = [None]
        NMSE_samp_FFT = [None]
        SNR_samp_FFT = [None]
        PCC_samp_FFT = [None]
        #
        MAE_samp_FFT_do = [None]
        NMSE_samp_FFT_do = [None]
        SNR_samp_FFT_do = [None]
        PCC_samp_FFT_do = [None]
        #
        MAE_samp_FFT_nodo = [None]
        NMSE_samp_FFT_nodo = [None]
        SNR_samp_FFT_nodo = [None]
        PCC_samp_FFT_nodo = [None]
        #

    if True:
        print(" ")
        print(f"(mn, std) MSE for {len(MSE_samp_EEG_sig)} all      samples of EEG: ({np.mean(MSE_samp_EEG_sig):0.5f}, {np.std(MSE_samp_EEG_sig):0.5f})")
        print(f"(mn, std) MSE for {len(MSE_samp_EEG_sig_do)} drop-out samples of EEG: ({np.nanmean(MSE_samp_EEG_sig_do):0.5f}, {np.nanstd(MSE_samp_EEG_sig_do):0.5f})")
        print(f"(mn, std) MSE for {len(MSE_samp_EEG_sig_nodo)} non-drop samples of EEG: ({np.nanmean(MSE_samp_EEG_sig_nodo):0.5f}, {np.nanstd(MSE_samp_EEG_sig_nodo):0.5f})")
        try:
            print(f"(mn, std) MSE for {len(MSE_samp_FFT)} all      samples of FFT: ({np.mean(MSE_samp_FFT):0.5f}, {np.std(MSE_samp_FFT):0.5f})")
            print(f"(mn, std) MSE for {len(MSE_samp_FFT_do)} drop-out samples of FFT: ({np.nanmean(MSE_samp_FFT_do):0.5f}, {np.nanstd(MSE_samp_FFT_do):0.5f})")
            print(f"(mn, std) MSE for {len(MSE_samp_FFT_nodo)} non-drop samples of FFT: ({np.nanmean(MSE_samp_FFT_nodo):0.5f}, {np.nanstd(MSE_samp_FFT_nodo):0.5f})")
        except:
            pass
        print(" ")

    return MSE_samp_EEG_sig, \
           MSE_samp_EEG_sig_do, \
           MSE_samp_EEG_sig_nodo, \
           MSE_samp_FFT, \
           MSE_samp_FFT_do, \
           MSE_samp_FFT_nodo, \
           MSE_samp_latent, \
           MSE_samp_EEG_pos, \
           MAE_samp_EEG_sig, \
           NMSE_samp_EEG_sig, \
           SNR_samp_EEG_sig, \
           PCC_samp_EEG_sig, \
           MAE_samp_EEG_sig_do, \
           NMSE_samp_EEG_sig_do, \
           SNR_samp_EEG_sig_do, \
           PCC_samp_EEG_sig_do, \
           MAE_samp_EEG_sig_nodo, \
           NMSE_samp_EEG_sig_nodo, \
           SNR_samp_EEG_sig_nodo, \
           PCC_samp_EEG_sig_nodo, \
           MAE_samp_FFT, \
           NMSE_samp_FFT, \
           SNR_samp_FFT, \
           PCC_samp_FFT, \
           MAE_samp_FFT_do, \
           NMSE_samp_FFT_do, \
           SNR_samp_FFT_do, \
           PCC_samp_FFT_do, \
           MAE_samp_FFT_nodo, \
           NMSE_samp_FFT_nodo, \
           SNR_samp_FFT_nodo, \
           PCC_samp_FFT_nodo, \
           MAE_samp_latent, \
           NMSE_samp_latent, \
           SNR_samp_latent, \
           PCC_samp_latent, \
           MAE_samp_EEG_pos, \
           NMSE_samp_EEG_pos, \
           SNR_samp_EEG_pos, \
           PCC_samp_EEG_pos


def plot_unwrapped_signals(model_signal_input_unwrapped, 
                            model_signal_output_unwrapped, 
                            eeg_signal_unwrapped, 
                            MSE_samp_EEG_sig,
                            #
                            model_position_input_unwrapped, 
                            model_position_output_unwrapped, 
                            MSE_samp_EEG_pos,
                            #
                            fft_signal_input_unwrapped, 
                            fft_signal_output_unwrapped,
                            MSE_samp_FFT,
                            MSE_samp_FFT_do,
                            MSE_samp_FFT_nodo,
                            #
                            latent_data_unwrapped,
                            latent_recon_unwrapped,
                            MSE_samp_latent,
                            #
                            fs,
                            freqs,
                            batch_cntr,
                            batch_idx,
                            dir_base,  
                            fname_suptag,
                            #
                            plot_eeg_signal_samples,
                            plot_eeg_position_samples,
                            plot_fft_samples,
                            plot_latent_samples,
                            args,
                            mne_interpolated_signals=None):

        """
        Plot original and reconstructed signals, channel positions, FFTs, latents for a single batch.
        """

        for samp in range(len(model_signal_input_unwrapped)):
            print(f"sample {samp}")

            # print(f"In plot_unwrapped_signals: How many dropped out chans?? What does model_signal_input_unwrapped[samp] & model_signal_output_unwrapped[samp] look like???")
            # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)

            # (1). Plot EEG time course for data and reconstruction on same axis (one ax per channel). One figure per sample.
            if plot_eeg_signal_samples:
                # 1a. Plot with non-dropout signal too.
                plot_compare_eeg_signal(data=model_signal_input_unwrapped[samp],
                                        reconst=model_signal_output_unwrapped[samp],
                                        mse_value=MSE_samp_EEG_sig[samp],
                                        eeg_signal=eeg_signal_unwrapped[samp],
                                        mne_reconstruction = mne_interpolated_signals[samp] if mne_interpolated_signals else None,
                                        fs=fs,
                                        batch=batch_cntr,
                                        sample=samp,
                                        idx=batch_idx[samp].item(),
                                        fname_tag=""+fname_suptag,
                                        dir_base=dir_base,
                )
                # 1b. plot without non-dropout signal.
                plot_compare_eeg_signal(data=model_signal_input_unwrapped[samp],
                                        reconst=model_signal_output_unwrapped[samp],
                                        mse_value=MSE_samp_EEG_sig[samp],
                                        # eeg_signal=eeg_signal_unwrapped[samp], # comment out to plot without non-dropped out data
                                        # mne_reconstruction = mne_interpolated_signals[samp] if mne_interpolated_signals else None,
                                        fs=fs, 
                                        batch=batch_cntr, 
                                        sample=samp,
                                        idx=batch_idx[samp].item(),
                                        fname_tag="_dropout"+fname_suptag,
                                        dir_base=dir_base,
                )

            # (2). Plot channel positions if we are concatenating channel {x,y,z} with EEG data and predicting them. Maybe Old.
            if plot_eeg_position_samples and args.data.cat_chan_xyz_and_eeg and args.data.dont_noise_chan_xyz:
                plot_compare_eeg_position(model_position_input_unwrapped[samp],
                                        model_position_output_unwrapped[samp],
                                        MSE_samp_EEG_pos[samp],
                                        batch=batch_cntr, 
                                        sample=samp,
                                        idx=batch_idx[samp].item(),
                                        fname_tag=""+fname_suptag,
                                        dir_base=dir_base,
                )


            # (3). Plot EEG FFT frequency specturms for data and reconstruction on same axis (one ax per channel). One figure per sample.
            if plot_fft_samples:
                plot_compare_fft(fft_signal_input_unwrapped[samp], 
                                fft_signal_output_unwrapped[samp],
                                MSE_samp_FFT[samp],
                                MSE_samp_FFT_do[samp],
                                MSE_samp_FFT_nodo[samp],
                                freqs=freqs, 
                                batch=batch_cntr,
                                sample=samp,
                                idx=batch_idx[samp].item(),
                                fname_tag=""+fname_suptag,
                                dir_base=dir_base,
                )

            # (4). Plot Latents encoder consistency computation.
            if plot_latent_samples:
                plot_compare_latents(latent_data_unwrapped[samp], 
                                    latent_recon_unwrapped[samp], 
                                    MSE_samp_latent[samp],
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
    print('In evaluate boo!')
    # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

    fs = args.data.sample_rate
    num_t = args.data.seq_len


    with ExitStack() as context_stack:
        ## tokenizer = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
        validate_train_args(
            args,
            # tokenizer.n_words,
        )

        if get_is_master():
            os.makedirs(args.dump_dir, exist_ok=True)
            dump_config(args, Path(args.dump_dir) / "config.yaml")
        init_logger(Path(args.dump_dir) / "train.log")
        init_signal_handler(set_preemption_flag)  # For handling preemption signals.
        setup_env(args.env)

        setup_torch_distributed(args.distributed)
        world_mesh = get_device_mesh(args.distributed)
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

        # Initializing Model in meta device allows us to initialize models much bigger than 1 gpu's memory
        
        if LOAD_THE_MODEL:
            with torch.device("meta"):
                model = EncoderDecoder(args.model)

            logger.info("Model is built !")

            model_param_count = get_num_params(model)


            # (CW) - DO NOT NEED TO SHARD MODEL FOR INFERENCE
            # model = parallelize_model(
            #     model,
            #     world_mesh,
            #     args.model,
            #     args.distributed,
            #     fsdp_grouping_plan=[],#build_fsdp_grouping_plan(args.model),
            #     # fsdp_grouping_plan=build_fsdp_grouping_plan(args.model),
            #     # tp_parallelize=tp_parallelize,
            #     no_recompute_ops=None,#get_no_recompute_ops(),
            # )

            # model = torch.compile(model)              # <-- this does not work. InductorError: LoweringException: AttributeError: 'Symbol' object has no attribute 'get_device'

            model.sample = torch.compile(model.sample)  # <-- this works. Why?!? The for loop in .sample causes graph breaks??
            model.encoder = torch.compile(model.encoder)

            # Once we shard the model on different gpus we can actually initialize the model
            # First we create empty tensors of the correct shapes
            model = model.to_empty(device=torch.cuda.current_device()) # Use local device, not cuda:0
            # Then we init the model. Please make sure this function initializes *ALL* parameters
            # and buffers, otherwise you will have random values in the unitialized tensors
            # which will silently fail (give nan gradients for example)


            # print(f"Before: {model.encoder.dropout_vec=}")

            # logger.info(f"!!!! Loading initial model from {args.checkpoint.init_ckpt_path} !!!! \n\n")
            # load_from_checkpoint(args.checkpoint.init_ckpt_path, model, model_key="model") 
            # logger.info("!!!!!!!!!!! Model loaded from checkpoint completed !!!!!!!!!!!")
            # check_model_value_range(model, range=10.0, std=1.0)
            ##(CW. Replace below with above)
            if args.checkpoint.init_ckpt_path:
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(args.model.seed)
                    model.init_weights()
                check_model_value_range(model, range=10.0, std=1.0)
                logger.info(f"!!!! Loading initial model from {args.checkpoint.init_ckpt_path} !!!! \n\n")
                load_from_checkpoint(args.checkpoint.init_ckpt_path, model, model_key="model") # Put model_key="" if its directly the model checkpoint
                # model.encoder.rope_embeddings.reset_parameters() # For RoPe initialization since it's a buffer it might not be loaded
                # model.decoder.rope_embeddings.reset_parameters() # For RoPe initialization since it's a buffer it might not be loaded
                # model.decoder.t_embedder.reset_parameters(args.model.init_base_std)
                logger.info("!!!!!!!!!!! Model loaded from checkpoint completed !!!!!!!!!!!")
                check_model_value_range(model, range=10.0, std=1.0)
            else:
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(args.model.seed)
                    model.init_weights()
            check_model_value_range(model, range=10.0, std=1.0)

            # log model size
            logger.info(f"Model size: {model_param_count:,} total parameters")

            gpu_memory_monitor = GPUMemoryMonitor("cuda")
            logger.info(
                f"GPU capacity: {gpu_memory_monitor.device_name} ({gpu_memory_monitor.device_index}) "
                f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
            )
            logger.info(f"GPU memory usage: {gpu_memory_monitor}")


            ## DONT THINK I NEED THIS. (CW)
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

            # print(f"After: {model.encoder.dropout_vec=}")

            # Make seed unique per GPU/rank by adding rank to base seed
            # dp_rank=0 # (CW) - hard code this. Not running data_parallel
            rank_seed = args.seed + dp_rank
            torch.manual_seed(rank_seed)
            torch.cuda.manual_seed(rank_seed)

            logger.info(f"Setting torch seed to {rank_seed} for rank {dp_rank}")
            
            # Also make numpy and random seeds unique per rank
            np.random.seed(rank_seed)
            random.seed(rank_seed)

            model.eval()
            metric_logger = context_stack.enter_context(
                MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args)
        )
        # data_loader = context_stack.enter_context(
        #     create_dataloader(
        #         args.data,
        #     )
        # )


        # print("After loading model from checkpoint, before creating dataloader.")
        # import IPython; print('\n\n\Debug:'); IPython.embed(); import time; time.sleep(0.3)

        # args.data.load_csv() (CW) - old audio stuff.
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

            eeg_sig_norm = 10.0 # (CW) - normalization factor for eeg signal.
            eeg_sig_clip = 1.0 #None  # (CW) - clipping factor for eeg signal.

            while True:
                epoch += 1
                logger.info(f"Starting epoch: {epoch}")
                for idx,batch in enumerate(dataloader):

                    # print(f"Inside make_batch_iterator: {batch.keys()=}, {batch['eeg_signal'].shape=}")
                    # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

                    eeg_signal = batch['eeg_signal']

                    eeg_signal = eeg_signal/eeg_sig_norm # (CW) - Divide by eeg_sig_norm to normalize the data and change its STD.

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

        torch_profiler = None #context_stack.enter_context(
        #     maybe_run_profiler(args.dump_dir, model, args.profiling)
        # )

        #make sure all model parameters require gradients
        if LOAD_THE_MODEL:
            for p in model.parameters():
                p.requires_grad = False # True (False for eval, True for training)

        data_processor = EEGProcessor(args.data).to(torch.cuda.current_device())

        # print(f"After loading in model")
        # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)


        # Trying to print out how information is flowing through the model.
        if False:
            for xxx in model.decoder.named_modules():
                print(f"{xxx=}")



        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #
        # (2). Here, run EEG data through autoencoder model and compare model input with model output
        #       TO DO: Compare dec_out to batch['encoder_input'] or eeg_signal.
        #

        plot_eeg_signal_samples = True      # Plot raw eeg for data and model reconstruction for single samples
        plot_eeg_position_samples = False #True   # Scatter eeg channel position GT vs reconstruction for single samples
        plot_fft_samples = False #True             # Plot fft of eeg for data and model reconstruction for single samples
        plot_latent_samples = False #True
        compute_encoder_consistency = True
        compute_reconstruction_metrics_stats_across_dataset = True

        sample_steps = 50    # for diffusion process in .sample - Default is 50
        cfg = 1.0            # for diffusion process in .sample - Default is 1.0 (i.e., no cfg)

        dir_base = 'figures/' + '/'.join(args.checkpoint.init_ckpt_path.split('/')[-3:]) + f'/cfg{cfg}'
        print(f"Saving output figures to: {dir_base=}")
        os.makedirs(dir_base, exist_ok=True)

        # Loop through batches of data from dataloader and gather up mean & std of data
        print_batch_stats = False
        if print_batch_stats:
            batch_mean = []
            batch_std = []
            batch_cntr = 0
            while True:
                batch = next(batch_iterator)     
                batch_cntr += 1
                print(f"{batch_cntr=}, {epoch=}")
                batch_mean.append( batch['eeg_signal'].mean().item() )
                batch_std.append( batch['eeg_signal'].std().item() )
                if epoch > 1 or batch_cntr > 20000:
                    break

            print(f"After {batch_cntr} batches through data loader:")
            print(f"Batch std: (mn, std) ({np.array(batch_std).mean()}, {np.array(batch_std).std()})")
            print(f"Batch mean: (mn, std) ({np.array(batch_mean).mean()}, {np.array(batch_mean).std()})")

            print(f"After Loop through batches of data from dataloader and gather up mean & std of data")
            import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)



        num_batches = 5
        batch_cntr = 0

        

        if compute_reconstruction_metrics_stats_across_dataset:
            MAE_samp_EEG_sig_do_list = []
            NMSE_samp_EEG_sig_do_list = []
            SNR_samp_EEG_sig_do_list = []
            PCC_samp_EEG_sig_do_list = []
            #
            MAE_samp_EEG_mne_do_list = []
            NMSE_samp_EEG_mne_do_list = []
            SNR_samp_EEG_mne_do_list = []
            PCC_samp_EEG_mne_do_list = []
            #
            MAE_samp_EEG_sig_nodo_list = []
            NMSE_samp_EEG_sig_nodo_list = []
            SNR_samp_EEG_sig_nodo_list = []
            PCC_samp_EEG_sig_nodo_list = []
            #
            MAE_samp_EEG_mne_nodo_list = [] 
            NMSE_samp_EEG_mne_nodo_list = []
            SNR_samp_EEG_mne_nodo_list = []
            PCC_samp_EEG_mne_nodo_list = []



        while True:
            batch = next(batch_iterator)     
            batch_cntr += 1

            # if batch_cntr < 3:
            #     continue

            # NOTE: Doing channel masking based on freq_mask input into EncoderDecoder.forward in transformer.py 
            mask_bad_chans = False # (CW) - There shouldnt be any more zero'ed out channels. 
            if mask_bad_chans:
                print(f"Masking out any bad (all zero) channels.")
                bad_mask = batch['eeg_signal'].abs().sum(axis=1)!=0 # mask out channels that are ALL ZERO --> freq_masks [B,1,C] that is [0 if bad, 1 if good]
                batch['freq_masks'] = bad_mask.unsqueeze(2).int() # [B,C,1]  

            eeg_signal = batch['eeg_signal']

            # batch_ids = batch.pop('ids', None)
            batch_idx = batch.pop('idx', None)
            batch_dataset_id = batch.pop('dataset_id', None)   # NOTE: pop takes them out of batch. (CW) - if left in, breaks things below and not training on these.
            
            # print(f"Making plots for raw vs reconstruction for {batch_idx=}, {batch_dataset_id=}") 

            # batch, loss_weights_batch = process_batch_data_compiled(batch, train_state.step) # (CW) - was this
            # batch, loss_weights_batch = process_batch_data(batch, train_state.step)   # - option 2 (CW) - maybe change to noncompiled version ???
            with torch.no_grad(): 
                batch = data_processor.process(**batch)                             #  > option 3. (CW)
            
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()} 

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

            # print(f"{args.model.tok_idx_type=} and {args.model.rope_dim=}")

            # print(f"Before model.sample")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)  

            with torch.no_grad():
                z, inference_at_step = model.sample(
                    encoder_input=batch['encoder_input'].unsqueeze(0),
                    seq_lens=batch['seq_lens'],
                    tok_idx=tok_idx,
                    cfg=cfg,
                    sample_steps=sample_steps,
                )    



            ## "Encoder Consistency": Compute MSE between latent representations encoder builds from raw-data and model reconstructions
            if compute_encoder_consistency:
                # Push reconstruction and original data back through encoder into latent space
                with torch.no_grad():
                    latent_data, _ = model.encoder(
                                            token_values=batch['encoder_input'].unsqueeze(0), 
                                            seq_lens=batch['seq_lens'],
                                            tok_idx=tok_idx,
                    )
                    latent_recon, _ = model.encoder(
                                            token_values=z, #z_masked, 
                                            seq_lens=batch['seq_lens'],
                                            tok_idx=tok_idx,
                    )
            else:
                latent_data, latent_recon = None, None

            # print(f"After drawing samples from model and getting latents from encoder.")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)  



            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     

            # model_output = z #.cpu().float().numpy() 
            # model_input = batch['encoder_input'] #.cpu().numpy()        # Includes channel dropout
            # eeg_signal = batch['eeg_signal'] #.cpu().numpy() # TO DO.   # Original eeg signal without channel dropout

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
                latent_data_unwrapped, \
                latent_recon_unwrapped, \
                t_coarse_unwrapped = unwrap_all_the_signals(model_output=z, 
                                                            latent_data=latent_data, 
                                                            latent_recon=latent_recon, 
                                                            batch=batch, 
                                                            args=args)    

                # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)  
                
                # eeg_signal_unwrapped: original data | is list, each item has ch*timepoints
                # model_signal_input_unwrapped: input with dropped channels 
                # model_signal_output_unwrapped, output of the model 
                # chan_pos = model_position_input_unwrapped[0].reshape(-1,tc,3)[:,0,:] #channel position requires this reshape
                

                #jm - Prepare channel positions for MNE
                chan_pos_list = [model_position_input_unwrapped[i].reshape(-1, tc, 3)[:, 0, :] for i in range(len(model_signal_input_unwrapped))]
                #jm - Apply MNE interpolation to dropped-out channels
                mne_interpolated_signals = interpolate_signals_with_mne(
                    signals=model_signal_input_unwrapped,
                    channel_positions=chan_pos_list,
                    sampling_rate=fs,
                    mark_zero_variance=True
                )

                # Compute FFT of signal input into model and signal output from model.
                fft_signal_input_unwrapped, freqs = compute_sig_FFT(eeg_signal_unwrapped, fs) # (CW) - non-dropped-out signal.
                fft_signal_output_unwrapped, _ = compute_sig_FFT(model_signal_output_unwrapped, fs)            

                # Compute reconstruction-based metrics between original and reconstructions from model
                MSE_samp_EEG_sig, \
                MSE_samp_EEG_sig_do, \
                MSE_samp_EEG_sig_nodo, \
                MSE_samp_FFT, \
                MSE_samp_FFT_do, \
                MSE_samp_FFT_nodo, \
                MSE_samp_latent, \
                MSE_samp_EEG_pos, \
                MAE_samp_EEG_sig, \
                NMSE_samp_EEG_sig, \
                SNR_samp_EEG_sig, \
                PCC_samp_EEG_sig, \
                MAE_samp_EEG_sig_do, \
                NMSE_samp_EEG_sig_do, \
                SNR_samp_EEG_sig_do, \
                PCC_samp_EEG_sig_do, \
                MAE_samp_EEG_sig_nodo, \
                NMSE_samp_EEG_sig_nodo, \
                SNR_samp_EEG_sig_nodo, \
                PCC_samp_EEG_sig_nodo, \
                MAE_samp_FFT, \
                NMSE_samp_FFT, \
                SNR_samp_FFT, \
                PCC_samp_FFT, \
                MAE_samp_FFT_do, \
                NMSE_samp_FFT_do, \
                SNR_samp_FFT_do, \
                PCC_samp_FFT_do, \
                MAE_samp_FFT_nodo, \
                NMSE_samp_FFT_nodo, \
                SNR_samp_FFT_nodo, \
                PCC_samp_FFT_nodo, \
                MAE_samp_latent, \
                NMSE_samp_latent, \
                SNR_samp_latent, \
                PCC_samp_latent, \
                MAE_samp_EEG_pos, \
                NMSE_samp_EEG_pos, \
                SNR_samp_EEG_pos, \
                PCC_samp_EEG_pos = compute_reconstruction_metrics_unwrapped_signals(model_signal_input_unwrapped, 
                                                                                    model_signal_output_unwrapped,  
                                                                                    eeg_signal_unwrapped, 
                                                                                    model_position_input_unwrapped, 
                                                                                    model_position_output_unwrapped, 
                                                                                    latent_data_unwrapped, 
                                                                                    latent_recon_unwrapped,
                                                                                    fft_signal_input_unwrapped,
                                                                                    fft_signal_output_unwrapped)


                # Compute reconstruction-based metrics between original and mne-linear-interpolated signals
                MSE_samp_EEG_mne, \
                MSE_samp_EEG_mne_do, \
                MSE_samp_EEG_mne_nodo, \
                MSE_samp_FFT_mne, \
                MSE_samp_FFT_mne_do, \
                MSE_samp_FFT_mne_nodo, \
                _, \
                _, \
                MAE_samp_EEG_mne, \
                NMSE_samp_EEG_mne, \
                SNR_samp_EEG_mne, \
                PCC_samp_EEG_mne, \
                MAE_samp_EEG_mne_do, \
                NMSE_samp_EEG_mne_do, \
                SNR_samp_EEG_mne_do, \
                PCC_samp_EEG_mne_do, \
                MAE_samp_EEG_mne_nodo, \
                NMSE_samp_EEG_mne_nodo, \
                SNR_samp_EEG_mne_nodo, \
                PCC_samp_EEG_mne_nodo, \
                MAE_samp_FFT_mne, \
                NMSE_samp_FFT_mne, \
                SNR_samp_FFT_mne, \
                PCC_samp_FFT_mne, \
                MAE_samp_FFT_mne_do, \
                NMSE_samp_FFT_mne_do, \
                SNR_samp_FFT_mne_do, \
                PCC_samp_FFT_mne_do, \
                MAE_samp_FFT_mne_nodo, \
                NMSE_samp_FFT_mne_nodo, \
                SNR_samp_FFT_mne_nodo, \
                PCC_samp_FFT_mne_nodo, \
                _, \
                _, \
                _, \
                _, \
                _, \
                _, \
                _, \
                _ = compute_reconstruction_metrics_unwrapped_signals(model_signal_input_unwrapped, 
                                                                     mne_interpolated_signals, 
                                                                     eeg_signal_unwrapped)



                # Plot signals
                # fname_suptag=""
                plot_unwrapped_signals(model_signal_input_unwrapped, 
                                        model_signal_output_unwrapped, 
                                        eeg_signal_unwrapped, 
                                        MSE_samp_EEG_sig,
                                        #
                                        model_position_input_unwrapped, 
                                        model_position_output_unwrapped, 
                                        MSE_samp_EEG_pos,
                                        #
                                        fft_signal_input_unwrapped, 
                                        fft_signal_output_unwrapped,
                                        MSE_samp_FFT,
                                        MSE_samp_FFT_do,
                                        MSE_samp_FFT_nodo,
                                        #
                                        latent_data_unwrapped,
                                        latent_recon_unwrapped,
                                        MSE_samp_latent,
                                        #
                                        fs,
                                        freqs,
                                        batch_cntr,
                                        batch_idx,
                                        dir_base,
                                        fname_suptag,  
                                        #
                                        plot_eeg_signal_samples,
                                        plot_eeg_position_samples,
                                        plot_fft_samples,
                                        plot_latent_samples,
                                        args,
                                        mne_interpolated_signals=mne_interpolated_signals)


            # print(f"After plotting signals")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)


            # Gather up all metrics across batches into bigger lists

            if compute_reconstruction_metrics_stats_across_dataset: 
                MAE_samp_EEG_sig_do_list.extend(MAE_samp_EEG_sig_do)   
                NMSE_samp_EEG_sig_do_list.extend(NMSE_samp_EEG_sig_do)
                SNR_samp_EEG_sig_do_list.extend(SNR_samp_EEG_sig_do)
                PCC_samp_EEG_sig_do_list.extend(PCC_samp_EEG_sig_do)
                #
                MAE_samp_EEG_mne_do_list.extend(MAE_samp_EEG_mne_do)
                NMSE_samp_EEG_mne_do_list.extend(NMSE_samp_EEG_mne_do)
                SNR_samp_EEG_mne_do_list.extend(SNR_samp_EEG_mne_do)
                PCC_samp_EEG_mne_do_list.extend(PCC_samp_EEG_mne_do)
                #
                MAE_samp_EEG_sig_nodo_list.extend(MAE_samp_EEG_sig_nodo)
                NMSE_samp_EEG_sig_nodo_list.extend(NMSE_samp_EEG_sig_nodo)
                SNR_samp_EEG_sig_nodo_list.extend(SNR_samp_EEG_sig_nodo)
                PCC_samp_EEG_sig_nodo_list.extend(PCC_samp_EEG_sig_nodo)
                #
                MAE_samp_EEG_mne_nodo_list.extend(MAE_samp_EEG_mne_nodo) 
                NMSE_samp_EEG_mne_nodo_list.extend(NMSE_samp_EEG_mne_nodo)
                SNR_samp_EEG_mne_nodo_list.extend(SNR_samp_EEG_mne_nodo)
                PCC_samp_EEG_mne_nodo_list.extend(PCC_samp_EEG_mne_nodo)




            # Here if you want to only do a certain number of batches (like for making a couple plots))
            if batch_cntr >= num_batches:
                break

            # # Here if you want to only do a certain number of epochs (like for computng eval metric stats)
            # if epoch > 1:
            #     break

        ## Display Stats of reconstruction-based metrics across batches of data
        try:
            print(f"\n\n{len(MAE_samp_EEG_mne_do_list)} samples from {data_loader.dataset.key_prefix} with channel dropout rate {args.data.channel_dropout_prob}") # backblaze path in EEGDataset_b2
        except:
            print(f"\n\n{len(MAE_samp_EEG_mne_do_list)} samples from {data_loader.dataset.memmap_paths[0].parts[5]} with channel dropout rate {args.data.channel_dropout_prob}") # local path in EEGDataset_v2

        print("\nMAE:")
        print(f"\tZUNA recon: (mean +/- std) ({np.array(MAE_samp_EEG_sig_do_list).mean():.4f} +/- {np.array(MAE_samp_EEG_sig_do_list).std():.4f})")
        print(f"\tmne interp: (mean +/- std) ({np.array(MAE_samp_EEG_mne_do_list).mean():.4f} +/- {np.array(MAE_samp_EEG_mne_do_list).std():.4f})")
        print("NMSE:")
        print(f"\tZUNA recon: (mean +/- std) ({np.array(NMSE_samp_EEG_sig_do_list).mean():.4f} +/- {np.array(NMSE_samp_EEG_sig_do_list).std():.4f})")
        print(f"\tmne  interp: (mean +/- std) ({np.array(NMSE_samp_EEG_mne_do_list).mean():.4f} +/- {np.array(NMSE_samp_EEG_mne_do_list).std():.4f})")
        print("SNR:")
        print(f"\tZUNA recon: (mean +/- std) ({np.array(SNR_samp_EEG_sig_do_list).mean():.4f} +/- {np.array(SNR_samp_EEG_sig_do_list).std():.4f})")
        print(f"\tmne interp: (mean +/- std) ({np.array(SNR_samp_EEG_mne_do_list).mean():.4f} +/- {np.array(SNR_samp_EEG_mne_do_list).std():.4f})")
        # print("PCC:") 
        # print(f"\tZUNA recon: (mean +/- std) ({np.array(PCC_samp_EEG_sig_do_list).mean():.4f} +/- {np.array(PCC_samp_EEG_sig_do_list).std():.4f})")
        # print(f"\tmne interp: (mean +/- std) ({np.array(PCC_samp_EEG_mne_do_list).mean():.4f} +/- {np.array(PCC_samp_EEG_mne_do_list).std():.4f})")
        print(f"\n\n")

        print("After looping over dataloader")
        import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)

#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#


def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        model: LMTransformerArgsgs

    @dataclass
    class LMTransformerArgsgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgsgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate TrainArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call train.py with train.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in TrainArgs dataclass.
    """
    cli_args = OmegaConf.from_cli()

    file_cfig = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfig = OmegaConf.structured(TrainArgs())
    cfig = OmegaConf.merge(default_cfig, file_cfig, cli_args)
    cfig = OmegaConf.to_object(cfig)

    # print(cfig)

    # print(f"I am in main after imports and after loading config, before diving into train.")
    # import IPython; print('\n\n Debug:'); IPython.embed(); import time;  time.sleep(0.3)

    evaluate(cfig)


if __name__ == "__main__":
    main()
