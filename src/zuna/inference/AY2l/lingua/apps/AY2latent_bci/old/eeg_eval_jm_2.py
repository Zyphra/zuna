
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


# 1st, setup tmux and docker with lingua.sh
#   >> bash /mnt/home/chris/workspace/AY2l/lingua/lingua.sh (on Crusoe)
#
# 2nd, run something like:
#   >> CUDA_VISIBLE_DEVICES=3 python3 apps/AY2latent_bci/eeg_eval.py config=apps/AY2latent_bci/configs/config_bci_eval.yaml

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
# from lingua.data import (
#     DataArgs,
#     PackTokensState,
#     build_dataloader_from_args,
#     init_dataloader_state_from_args,
# )

# from apps.AY2latent.data import (
#     AudioDatasetArgs,
#     AutoencoderDataset,
#     create_dataloader
# )

from apps.AY2latent_bci.eeg_data import EEGDataset_v2, EEGProcessor, BCIDatasetArgs, create_dataloader
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
from torch._dynamo.decorators import mark_static_address
import functools

#MOABB PIPELINES
import numpy as np
import torch

from scipy.signal import welch
from scipy.stats import entropy, skew, kurtosis

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

logger = logging.getLogger()


LOAD_THE_MODEL = True # Flag to load model onto GPU or not. If False, just explore data.
fs = 512        # 256  # Hz - Sampling frequency of the EEG data
num_t = 2560    # 1280 - Number of time points in 5 seconds at 512 sample rate
MOABB = True

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

    print(f"Dawgg, process_batch_data")
    import IPython; print('\n\n Debug:'); IPython.embed(); import time;  time.sleep(0.3)

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

    args.model.max_seqlen = 4096 # (CW) int(args.data.sample_duration_seconds * args.data.sample_rate)
                                 # this needs to be bigger than 1280*1.5 to include registers.

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


def compute_eeg_PSD(eeg, fs, window_sec=1):
    """
    Computes band power per channel, per time window.

    Parameters:
        eeg (np.ndarray): shape [time, channels]
        fs (int): sampling rate
        window_sec (int): size of each time window in seconds

    Returns:
        np.ndarray: shape [num_windows, num_bands, num_channels]
    """
    bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta":  (13, 30),
        "Gamma": (30, 128),
    }
    band_names = list(bands.keys())
    num_bands = len(bands)

    time_len, num_chans = eeg.shape
    print(f"Are these shapes correct? : {time_len=} & {num_chans=}")

    window_samples = fs * window_sec
    num_windows = time_len // window_samples

    band_power = np.zeros((num_windows, num_bands, num_chans))

    for win in range(num_windows):
        start = win * window_samples
        end = start + window_samples
        window_data = eeg[start:end, :]  # shape: [window_samples, channels]

        for ch in range(num_chans):
            freqs, psd = welch(window_data[:, ch], fs=fs, nperseg=window_samples)

            for b_idx, (band, (low, high)) in enumerate(bands.items()):
                idx = np.logical_and(freqs >= low, freqs <= high)
                band_power[win, b_idx, ch] = np.trapezoid(psd[idx], freqs[idx])

    # Normalize each channel so that total power in all bands adds to 1.
    try:
        band_power_norm = band_power / band_power.sum(axis=1, keepdims=True) # quick hack to avoid divby0 error
    except:
        band_power_norm = band_power

    return band_power_norm, bands  # shape: [num_windows, num_bands, num_chans]


def compute_plv_matrix(data, lags=[0]):
    """
    Compute Phase Locking Value (PLV) matrix at different time lags.

    Parameters:
        data: (np.ndarray): shape [time, channels]
        lags: list of int (lags in samples)

    Returns:
        plv_matrix: np.ndarray [channels, channels]
        lag_matrix: np.ndarray [channels, channels] (lag that gave max PLV)
    """

    data = data.T # easiest change. --> np.ndarray [channels, time]

    analytic_signal = hilbert(data, axis=1)
    phases = np.angle(analytic_signal)  # shape: [channels, time]

    n_channels, n_time = phases.shape
    plv_matrix = np.zeros((n_channels, n_channels))
    lag_matrix = np.zeros((n_channels, n_channels), dtype=int)

    for i in range(n_channels):
        for j in range(i, n_channels):
            max_plv = 0.0
            best_lag = 0
            for lag in lags:
                if lag > 0:
                    phi_i = phases[i, :-lag]
                    phi_j = phases[j, lag:]
                elif lag < 0:
                    phi_i = phases[i, -lag:]
                    phi_j = phases[j, :lag]
                else:
                    phi_i = phases[i]
                    phi_j = phases[j]

                min_len = min(len(phi_i), len(phi_j))
                if min_len == 0:
                    continue

                phase_diff = phi_i[:min_len] - phi_j[:min_len]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))

                if plv > max_plv:
                    max_plv = plv
                    best_lag = lag

            plv_matrix[i, j] = max_plv
            plv_matrix[j, i] = max_plv  # symmetric

            lag_matrix[i, j] = best_lag
            lag_matrix[j, i] = -best_lag  # symmetric (reverse lag)

    return plv_matrix, lag_matrix


def plot_compare_eeg(data,
                     reconst,  
                     fs=256,
                     batch=0, 
                     sample=0,
                     idx=0,
                     fname_tag="",
                     dir_base="figures"):
    """
    (1). Plot EEG time trace (data & reconst), each channel on a different subplot.
    (2). Plot FFT spectrum (data & reconst), each channel on a different subplot.
    """
    assert data.shape == reconst.shape

    num_t, chans = data.shape
    print(f"Are these shapes correct? : {num_t=} & {chans=}")

    t = np.arange(num_t) / fs

    fig, axes = plt.subplots(8, 8, figsize=(24, 12))

    invert = False # This should be False!
    if invert:
        reconst = -reconst

    # print("Inside plot_compare_eeg")
    # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)   

    # Loop through each subplot and plot something
    ch=-1
    for i in range(8):
        for j in range(8):
            ch+=1

            # Plot time-domain EEG (offset by channel index)
            axes[i, j].plot(t, data[:, ch], "b-", linewidth=0.5, alpha=0.4)
            axes[i, j].plot(t, reconst[:, ch], "r-", linewidth=0.5, alpha=0.4)
            axes[i, j].set_xlim(t[0],t[-1])
            # axes[i, j].set_ylim(-5,5)
            axes[i, j].tick_params(axis='x', labelsize=10)
            axes[i, j].tick_params(axis='y', labelsize=10)
            axes[i, j].grid(True)
            axes[i, j].text(.98, .98, f"Ch{ch+1}", transform=axes[i, j].transAxes, ha='right', va='top', fontsize=12, color='black')
        
            if i==7 and j==0:
                axes[i, j].set_xlabel("Time (s)")
                axes[i, j].set_ylabel("Amp")
        
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # leave some space at top for suptitle
    fig.text(0.27, 0.97, "data", ha='center', va='center', fontsize=16, fontweight='bold', color='blue')
    fig.text(0.30, 0.97, "vs.", ha='center', va='center', fontsize=16, fontweight='bold', color='black')
    fig.text(0.34, 0.97, "reconst", ha='center', va='center', fontsize=16, fontweight='bold', color='red')
    if invert:
        fig.text(0.70, 0.97, "*reconst inverted", ha='left', va='center', fontsize=16, fontweight='bold', color='black')
    plt.suptitle(f"EEG - (batch={batch}, idx={idx}, sample={sample})", fontsize=16, fontweight='bold')

    plt.savefig(f"{dir_base}/eeg_compare_B{batch}_S{sample}{fname_tag}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_compare_fft(data, 
                     reconst, 
                     freqs, 
                     batch=0, 
                     sample=0,
                     idx=0,
                     fname_tag="",
                     dir_base="figures"):
    
    """
    (1). Plot EEG time trace (data & reconst), each channel on a different subplot.
    (2). Plot FFT spectrum (data & reconst), each channel on a different subplot.
    """

    assert data.shape == reconst.shape

    fig, axes = plt.subplots(8, 8, figsize=(24, 12))

    # Loop through each subplot and plot something
    ch=-1
    for i in range(8):
        for j in range(8):
            ch+=1

            # Plot time-domain EEG (offset by channel index)
            axes[i, j].plot(freqs, data[:, ch], "b-", linewidth=0.5, alpha=0.4)
            axes[i, j].plot(freqs, reconst[:, ch], "r-", linewidth=0.5, alpha=0.4)
            axes[i, j].set_xlim(freqs[0],freqs[-1])
            # axes[i, j].set_ylim(-5,5)
            axes[i, j].tick_params(axis='x', labelsize=10)
            axes[i, j].tick_params(axis='y', labelsize=10)
            axes[i, j].grid(True)
            axes[i, j].text(.98, .98, f"Ch{ch+1}", transform=axes[i, j].transAxes, ha='right', va='top', fontsize=12, color='black')
            
            if i==7 and j==0:
                axes[i, j].set_xlabel("Freq (hz)")
                axes[i, j].set_ylabel("Amp")
        
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # leave some space at top for suptitle
    fig.text(0.27, 0.97, "data", ha='center', va='center', fontsize=16, fontweight='bold', color='blue')
    fig.text(0.30, 0.97, "vs.", ha='center', va='center', fontsize=16, fontweight='bold', color='black')
    fig.text(0.34, 0.97, "reconst", ha='center', va='center', fontsize=16, fontweight='bold', color='red')
    plt.suptitle(f"EEG FFT - (batch={batch}, idx={idx}, sample={sample})", fontsize=16, fontweight='bold')

    plt.savefig(f"{dir_base}/fft_compare_B{batch}_S{sample}{fname_tag}.png", dpi=300, bbox_inches='tight')
    plt.close()


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

def plot_eeg_fft_psd_plv(signal, 
                         fs=256, 
                         eeg_PSD = None, 
                         band_PSD = None, 
                         PLV_chans = None, 
                         PLV_best_lags = None, 
                         batch=0, 
                         sample=0,
                         idx=0,
                         fname_tag="",
                         dir_base="figures"):
    """
    Plot EEG signal on top,
    and FFT of EEG on bottom,
    and PSD for each channel & band on bottom,
    and PLV for each channel-pair somewhere else.
    """

    num_t, chans = signal.shape
    t = np.arange(num_t) / fs
    colors = get_distinct_colors(chans)

    # Setup axes for figure
    fig = plt.figure(figsize=(21, 10))

    plt.suptitle(f"EEG Eval - ({batch=},{idx=},{sample=})", fontsize=16, fontweight='bold')

    #                    [   x,   y, width, height]
    axEEG = fig.add_axes([0.01, 0.64, 0.45, 0.20])  
    axFFT = fig.add_axes([0.01, 0.32, 0.45, 0.20])
    axPSD = fig.add_axes([0.01, 0.01, 0.45, 0.20])
    axPLV = fig.add_axes([0.40, 0.01, 0.65, 0.90])

    # Plot time-domain EEG (offset by channel index)
    for ch in range(chans):
        axEEG.plot(t, signal[:, ch], color=colors[ch], alpha=0.2)
    axEEG.set_title(f"EEG Time-Domain")
    axEEG.set_xlabel("Time (s)")
    # axEEG.set_ylabel("Amplitude")
    # axEEG.axis('tight')
    axEEG.set_xlim(t[0],t[-1])
    axEEG.grid(True)

    # Plot frequency-domain (FFT)
    freqs = rfftfreq(num_t, 1/fs)
    for ch in range(chans):
        fft_vals = np.abs(rfft(signal[:, ch]))
        axFFT.plot(freqs, fft_vals, color=colors[ch], alpha=0.2)
    axFFT.set_title("EEG Frequency-Domain (FFT)")
    axFFT.set_xlabel("Frequency (Hz)")
    # axFFT.set_ylabel("Magnitude")
    # axFFT.axis('tight')
    axFFT.set_xlim(0,fs//2)
    axFFT.grid(True)

    # Imshow Power Spectral Density - (band v. channel)
    if eeg_PSD is None and band_PSD is None:
        eeg_PSD, band_PSD = compute_eeg_PSD(signal, fs, 5)

    imshow_PSD_ax(eeg_PSD, band_PSD, fig=fig, ax=axPSD)

    # Imshow Phase Locking Value - (channel v. channel)
    if PLV_chans is None and PLV_best_lags is None:
        max_lag = 9
        PLV_lags = list(range(-max_lag,max_lag+1)) #[0]
        PLV_chans, PLV_best_lags = compute_plv_matrix(signal, PLV_lags)

    imshow_PLV_ax(PLV_chans, PLV_best_lags, fig=fig, ax=axPLV)

    plt.tight_layout()
    plt.savefig(f"{dir_base}/eeg_sample_B{batch}_S{sample}{fname_tag}.png", dpi=300, bbox_inches='tight')
    plt.close()


def imshow_PLV_ax(PLV_chans, PLV_best_lags, fig, ax):
    """
    imshow Channel x Channel Phase Locking Values in matrix
    Show best lag in text over matrix entry.
    """

    chans = PLV_chans.shape[0]
    imPLV = ax.imshow(PLV_chans, cmap="turbo", interpolation='nearest', vmin=0, vmax=1)

    # Add text annotations (offset for contrast)
    for i in range(chans):
        for j in range(chans):
            val = PLV_best_lags[i, j].item()
            ax.text(j, i, f"{val}", ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
            ax.text(j+0.04, i+0.04, f"{val}", ha='center', va='center',
                    fontsize=8, color='black')

    # Axis and colorbar styling
    ax.set_title(f"Phase Locking Value (and Lag)", fontsize=14)
    ax.set_xlabel("Channel", fontsize=12)
    ax.set_ylabel("Channel", fontsize=12)
    ax.set_xticks(np.arange(chans))
    ax.set_yticks(np.arange(chans))
    ax.set_xticklabels([f"{i}" for i in range(chans)], fontsize=10, rotation=45)
    ax.set_yticklabels([f"{i}" for i in range(chans)], fontsize=10, rotation=0)
    cbar = fig.colorbar(imPLV, ax=ax, orientation='vertical')
    cbar.set_label('PLV & PSD', fontsize=16)


def imshow_PSD_ax(eeg_PSD, band_PSD, fig, ax):
    """
    imshow Power Spectral Density (chan vs band) on ax on fig.
    """
    imPSD = ax.imshow(eeg_PSD.squeeze(0), cmap='turbo', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(f"Power Spectral Density")
    # ax.set_ylabel("Freq Band")
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels([f"{k}\n{v} Hz" for k, v in band_PSD.items()], rotation=45)
    ax.set_xlabel("channel")
    # #
    # cax = inset_axes(ax,
    #                 width="1%",  
    #                 height="75%",
    #                 loc='upper right',
    #                 borderpad=4)
    # #
    # cbar = fig.colorbar(imPSD, cax=cax, ax=ax)
    # cbar.ax.yaxis.set_tick_params(color='white')  # tick marks
    # plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')  # tick labels
    # cbar.set_label("Power", color='white')  # colorbar label



#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#


def evaluate(args: TrainArgs):
    print('In evaluate boo!')

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
            # if True:

                # print(f"Loading the model")
                # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)

                model = EncoderDecoder(args.model)
                # 
                # model = model.to("cuda")
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
            model.sample = torch.compile(model.sample)  # <-- this works. Why?!?

            # Once we shard the model on different gpus we can actually initialize the model
            # First we create empty tensors of the correct shapes
            model = model.to_empty(device="cuda") # (CW) -  was this.
            # Then we init the model. Please make sure this function initializes *ALL* parameters
            # and buffers, otherwise you will have random values in the unitialized tensors
            # which will silently fail (give nan gradients for example)

            

            # print(f"After not parallelize model")
            # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)  


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
        # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)

        # args.data.load_csv() (CW) - old audio stuff.
        print("Entering create dataloader on rank", dp_rank)
        data_loader = create_dataloader(args.data, args.seed, dp_rank)
        print("Finishing create dataloader on rank", dp_rank)

        epoch = 0
        def make_batch_iterator(dataloader):  # (CW) Use with IterableDataset.
            nonlocal epoch
            # dataloader.sampler.set_epoch(epoch)
            print("Creating batch iterator of dataloader with length", len(dataloader), "and dataset of length", len(dataloader.dataset))
            while True:
                epoch += 1
                logger.info(f"Starting epoch: {epoch}")
                for idx,batch in enumerate(dataloader):

                    # if False:
                    #     # (CW) Here, perform channel norm to subtract off DC offset/.mean() and divide by .std()
                    #     print(f"Doing channel norm hack. Take this out for the new cleaned dataset.")
                    #     batch, chan_mu, chan_sig = ChannelNorm(batch, verbose=False) # TOOK THIS OUT FOR NEW DATA!

                    # # (CW) - Use batchnorm here to Normalize each batch to be mean0, std1. 
                    # bn = torch.nn.BatchNorm1d(num_features=64)
                    # batch = bn(batch)

                    eeg_signal = reshape_eeg_signal(batch['eeg_signal'])/5 # (CW) - HARDCODING dividing by 5 to normalize the data to be in range [-1,1] for training.

                    if True:
                        print("Clipping input at +/-1")
                        eeg_signal.clamp(min=-1, max=1) # NOTE: THIS NEEDS TO MATCH HOW MODEL WAS TRAINED!!!

                    ids = batch['ids'].float().mean().item() # which dataset data is coming from

                    # extract mode of dataset id from all samples. 
                    values, counts = torch.unique(batch['dataset_id'], return_counts=True)
                    mode_idx = torch.argmax(counts)
                    dataset_id_mode = values[mode_idx].item()

                    # yield batch # <- was this
                    # yield {"eeg_signal": batch, "chan_mu": chan_mu, "chan_sig": chan_sig, "idx": idx} # (CW) - because later expects data to be in a dict. When Chan Norming.
                    yield {"eeg_signal": eeg_signal, "ids": ids, "idx": idx, "dataset_id_mode": dataset_id_mode} # (CW) - because later expects data to be in a dict. No Longer Channel Norming with new cleaner data.

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


        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #
        # (2). Here, run EEG data through autoencoder model and compare model input with model output
        #       TO DO: Compare dec_out to batch['encoder_input'] or eeg_signal.
        #
        
        if MOABB: 
            #CHRIS, we'll need these two lines
            model_output_all = []
            model_input_all = []

        while True:
            dir_base = 'figures/' + '/'.join(args.checkpoint.init_ckpt_path.split('/')[-3:])
            os.makedirs(dir_base, exist_ok=True)

            batch = next(batch_iterator)


            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # 
            #
            # NOTE: Doing channel masking based on freq_mask input into EncoderDecoder.forward in transformer.py 
            mask_bad_chans = True 
            if mask_bad_chans:
                print(f"Masking out any bad (all zero) channels.")
                bad_mask = batch['eeg_signal'].abs().sum(axis=1)!=0 # mask out channels that are ALL ZERO --> freq_masks [B,1,C] that is [0 if bad, 1 if good]
                batch['freq_masks'] = bad_mask.unsqueeze(1).int()
            #
            #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

            eeg_signal = batch['eeg_signal']
            batch_idx = batch.pop('idx', None)                      # NOTE: pop takes them out of batch. (CW) - if left in, breaks things below and not training on these.
            batch_ids = batch.pop('ids', None)                      # NOTE: pop takes them out of batch. (CW) - if left in, breaks things below and not training on these.
            # batch_chan_mu = batch.pop('chan_mu', None)              # Not recording in WandB yet. 
            # batch_chan_sig = batch.pop('chan_sig', None)            # Not recording in WandB yet. Note: These dont exist if not doing ChanNorm, but None catches non-existence.
            batch_dataset_id = batch.pop('dataset_id_mode', None)  # NOTE: pop takes them out of batch. (CW) - if left in, breaks things below and not training on these.
             

            print(f"Making plots for raw vs reconstruction for {batch_idx=}, {batch_ids=}, {batch_dataset_id=}") #  

            # print(f"Look at batch of data")
            # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)

            # batch, loss_weights_batch = process_batch_data_compiled(batch, train_state.step) # (CW) - was this
            # batch, loss_weights_batch = process_batch_data(batch, train_state.step)   # - option 2 (CW) - maybe change to noncompiled version ???

            # with torch.no_grad():      
            #     batch = data_processor.process(**batch)                                 #  > option 3. (CW)
            #     dec_out, enc_loss, dec_loss = model(**batch)    

            with torch.no_grad(): 
                batch = data_processor.process(**batch)  
            
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()} 

            # print(f"Before model.sample(eeg_signal)")
            # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)   

            # with torch.no_grad(): 
            #     _, enc_loss, dec_loss = model(**batch) 
            # print(f"MOOOOOOOOO!   After model.forward()")
            # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)  


            with torch.no_grad():
                z, outs = model.sample(batch['encoder_input'])    

            # print(f"After model.sample()")
            # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)              

            model_output = z.cpu().float().numpy() 
            model_input = batch['encoder_input'].cpu().numpy()   
            
            if MOABB: 
                model_output_all.append(model_output)
                model_input_all.append(model_input)

            # # Compute MSE between model_input & model_output (inverted or not) for whole batch.
            # if True:
            #     MSE = np.abs(model_input - model_output).mean() # mean square error btwn data and reconst
            #     MSEi = np.abs(model_input + model_output).mean() # mean square error btwn data and inverted reconst
            #     #
            #     # Compute MSE between model_input & random normal null signal
            #     null_test  = np.random.randn(*model_output.shape) * 3 # N(0,3)
            #     MSEn = np.abs(model_input - null_test).mean() # mean square error btwn data and null random normal signal
            #     MSEni = np.abs(model_input + null_test).mean() # mean square error btwn data and null random normal signal

            #     print(f"MSE (data-x) = x=[reconst, inverted_reconst, null_signal] = {MSE:0.2f}, {MSEi:0.2f}, {MSEn:0.2f}")



            # # Compute FFT on model_input & model_output and compute MSE on those. No need to worry about inverting model_output
            # if True: 
            #     # Plot frequency-domain (FFT)
            #     freqs = rfftfreq(num_t, 1/fs)

            #     fft_data = np.abs(rfft(model_input, axis=1))
            #     data_norms = np.linalg.norm(fft_data, axis=1) 
            #     fft_data = fft_data / data_norms[:, np.newaxis, :]

            #     fft_reconst = np.abs(rfft(model_output, axis=1))
            #     reconst_norms = np.linalg.norm(fft_reconst, axis=1) 
            #     fft_reconst = fft_reconst / reconst_norms[:, np.newaxis, :]

            #     MSEf = np.abs(fft_data - fft_reconst).mean() # mean square error btwn data and reconst FFTs

            #     print(f"MSE (FFTdata - FFTreconst) = {MSEf:0.2f}")

            # # print("After model(batch), Compare dec_out to batch['encoder_input'] or eeg_signal.")
            # # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)

            # for samp in range(args.data.batch_size):

            #     # Visualize EEG, FFT, PSD & PLV for one sample on one figure for data and on one figure for reconst.
            #     if False:
            #         plot_eeg_fft_psd_plv(model_output[samp], 
            #                             fs=fs, 
            #                             batch=0, 
            #                             sample=samp,
            #                             idx=batch_idx,
            #                             fname_tag="_model",
            #                             dir_base=dir_base
            #         )
            #         #
            #         plot_eeg_fft_psd_plv(model_input[samp], 
            #                             fs=fs, 
            #                             batch=0, 
            #                             sample=samp,
            #                             idx=batch_idx,
            #                             fname_tag="_data",
            #                             dir_base=dir_base
            #         )


            #     # Plot EEG time course for data and reconstruction on same axis (one ax per channel). One figure per sample.
            #     if True:
            #         plot_compare_eeg(model_input[samp], 
            #                          model_output[samp], 
            #                          fs=fs, 
            #                          batch=batch_idx, 
            #                          sample=samp,
            #                          idx=batch_idx,
            #                          fname_tag="",
            #                          dir_base=dir_base
            #         )

            #     # Plot EEG FFT frequency specturms for data and reconstruction on same axis (one ax per channel). One figure per sample.
            #     if True:
            #         plot_compare_fft(fft_data[samp], 
            #                          fft_reconst[samp], 
            #                          freqs=freqs, 
            #                          batch=batch_idx, 
            #                          sample=samp,
            #                          idx=batch_idx,
            #                          fname_tag="",
            #                          dir_base=dir_base
            #         )

            if batch_idx > 10:
                if not MOABB: 
                    break
                if MOABB: 
                    #CHRIS, we'll need this code below 
                    model_input_all = model_input_all.detach().cpu().numpy()
                    model_output_all = model_output_all.detach().cpu().numpy()
                    from eeg_feature_clf import run_feature_clf

                    acc_in, acc_out = run_feature_clf(model_input_all,
                                                    model_output_all,
                                                    data_dir=args.data.data_dir,
                                                    fs=512)

                    print(f"Accuracy (features from raw input)  : {acc_in:.3f}")
                    print(f"Accuracy (features from autoencoder): {acc_out:.3f}")


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

    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(TrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    # print(cfg)

    # print(f"I am in main after imports and after loading config, before diving into train.")
    # import IPython; print('\n\n Debug:'); IPython.embed(); import time;  time.sleep(0.3)

    print(f"{fs=}")

    evaluate(cfg)


if __name__ == "__main__":
    main()
