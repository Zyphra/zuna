
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


# 1st, setup tmux and docker with lingua.sh
#   >> bash /data/groups/bci/chris/workspace/AY2l/lingua/lingua.sh 
#
# 2nd, run something like:
#   >> CUDA_VISIBLE_DEVICES=1 python3 apps/AY2latent_bci/eeg_eval.py config=apps/AY2latent_bci/configs/config_bci_eval.yaml
#   >> TORCH_LOGS="recompiles" CUDA_VISIBLE_DEVICES=1 python3 apps/AY2latent_bci/eeg_eval.py config=apps/AY2latent_bci/configs/config_bci_eval.yaml  2>&1 | tee /tmp/recompile_zuna.log


import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt


import gc
import json
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load
import logging
import os
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import random
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed
from torch.optim import lr_scheduler
from torch.distributed.checkpoint.stateful import Stateful

from lingua.args import dataclass_from_dict
from lingua.checkpoint import CheckpointArgs, CheckpointManager, load_from_checkpoint

from utils_pt_mne import interpolate_signals_with_mne

from apps.AY2latent_bci.eeg_data import (
    EEGProcessor,
    BCIDatasetArgs,
    create_dataloader_v2,
    # chop_and_reshape_signals, # for debug
    invert_reshape_signals,
    FifReconstructor,
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

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger()

LOAD_THE_MODEL = True           # Flag to load model onto GPU or not. If False, just explore data.



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
    """Scalar Pearson r between two arrays of identical shape."""
    a = np.asarray(y_true).ravel()
    b = np.asarray(y_pred).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    if denom == 0:
        return np.nan
    return float((a @ b) / denom)




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

    # mark the per-document dim of seq_lens/max_tc dynamic (see _maybe_mark_dynamic_ndoc)
    # so varying #packed-docs/users does NOT recompile the encoder/decoder + mask builders or
    # the @torch.compile'd EEGProcessor.process.
    dynamic_seq_lens: bool = True

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

    # Inference / diffusion sampling (supplied by the pipeline at eval time)
    diffusion_cfg: float = 1.0
    diffusion_sample_steps: int = 50
    plot_eeg_signal_samples: bool = True #False
    inference_figures_dir: str = "./inference_figures"



@dataclass
class TrainState(Stateful):
    step: int  # Nb of steps taken by the optimizer
    acc_step: int  # Nb of gradient-accumulation micro-steps (referenced by state_dict/load_state_dict + constructor)
    scheduler: lr_scheduler.LambdaLR
    # data_loader_state: PackTokensState

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

preemption_flag = dict(flag=False)

def set_preemption_flag(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Preemption ! checkpointing asap and exiting.")
    preemption_flag["flag"] = True






def plot_compare_eeg_signal(data,
                            reconst,  
                            mse_value,
                            pcc_value,
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

    # More general way of dropout - assumes anything that is 0.0 has been dropped.
    pct_dropout = (data==0).sum()/data.size
    where_dropout = data==0

    # Replaced MSE with NMSE.
    if eeg_signal is not None:
        MSE_dropout = compute_nmse(eeg_signal[where_dropout], reconst[where_dropout])
        PCC_dropout = compute_pcc(eeg_signal[where_dropout], reconst[where_dropout])
        MSE_nondrop = compute_nmse(eeg_signal[~where_dropout], reconst[~where_dropout])
        PCC_nondrop = compute_pcc(eeg_signal[~where_dropout], reconst[~where_dropout])
    if mne_reconstruction is not None:
        MSE_mne_dropout = compute_nmse(mne_reconstruction[where_dropout], reconst[where_dropout])
        PCC_mne_dropout = compute_pcc(mne_reconstruction[where_dropout], reconst[where_dropout])

    if dimx==dimy==1:
        # Single-channel case: (copy-pasted-edited from multi-chan below).
        ## KINDA BECOMING DEPRECATED ...
        ch=0
        axes.plot(t, data[:, ch], "b-", linewidth=0.5, alpha=0.4)
        axes.plot(t, reconst[:, ch], "r-", linewidth=0.5, alpha=0.4)
        if eeg_signal is not None:
            axes.plot(t, eeg_signal[:, ch], "g-", linewidth=0.5, alpha=0.4)
        if mne_reconstruction is not None:
            axes.plot(t, mne_reconstruction[:, ch], linestyle="-", color="grey", linewidth=0.5, alpha=0.4)
        axes.set_xlim(t[0],t[-1])
        axes.tick_params(axis='x', labelsize=10)
        axes.tick_params(axis='y', labelsize=10)
        axes.grid(True)
        axes.text(.98, .98, f"Ch{ch+1}", transform=axes.transAxes, ha='right', va='top', fontsize=12, color='black')
        axes.set_xlabel("Time (samples)")
        axes.set_ylabel("Amp")

    else:
        # Multi-channel case: Loop through each subplot and plot something
        ch=-1
        for i in range(dimx):
            for j in range(dimy):
                try:
                    ch+=1
                    ax = axes[i, j]

                    # Shade the dropped out channels and sections with light grey box.
                    parts = []
                    parts.append(reconst[:, ch])
                    if np.abs(data[:,ch]).sum() > 0: # if we havent dropped out whole channel
                        parts.append(data[:, ch])
                    if eeg_signal is not None:
                        parts.append(eeg_signal[:, ch])
                    if mne_reconstruction is not None:
                        parts.append(mne_reconstruction[:, ch])
                    y_lo = min(p.min() for p in parts)
                    y_hi = max(p.max() for p in parts)
                    if y_lo == y_hi:
                        y_lo, y_hi = y_lo - 0.5, y_hi + 0.5
                    mask = where_dropout[:, ch]
                    ax.fill_between(
                        t, y_lo, y_hi, where=mask,
                        color="lightgrey", alpha=0.45, linewidth=0, zorder=0,
                    )

                    # Plot time-domain EEG (offset by channel index)
                    if np.abs(data[:,ch]).sum() > 0: # if we havent dropped out whole channel
                        axes[i, j].plot(t, data[:, ch], "b-", linewidth=0.5, alpha=0.4)
                    axes[i, j].plot(t, reconst[:, ch], "r-", linewidth=0.5, alpha=0.4)
                    if eeg_signal is not None: # and where_dropout[ch]:
                        axes[i, j].plot(t, eeg_signal[:, ch], "g-", linewidth=0.5, alpha=0.4)
                    if mne_reconstruction is not None: # and where_dropout[ch]:
                        axes[i, j].plot(t, mne_reconstruction[:, ch], linestyle="-", color="grey", linewidth=0.5, alpha=0.4)
                    axes[i, j].set_xlim(t[0],t[-1])
                    axes[i, j].tick_params(axis='x', labelsize=10)
                    axes[i, j].tick_params(axis='y', labelsize=10)
                    axes[i, j].grid(True)
                    if True: #where_dropout[ch]:
                        axes[i, j].text(.98, .98, f"Ch{ch+1}", transform=axes[i, j].transAxes, ha='right', va='top', fontsize=12, color='black') #color='green')
                    else:
                        axes[i, j].text(.98, .98, f"Ch{ch+1}", transform=axes[i, j].transAxes, ha='right', va='top', fontsize=12, color='blue')

                    if i==(dimx-1) and j==0:
                        axes[i, j].set_xlabel("Time (samples)")
                        axes[i, j].set_ylabel("Amp")

                except:
                    break # If we run out of channels, just break
        
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # leave some space at top for suptitle
    fig.text(0.05, 0.97, "raw", ha='center', va='center', fontsize=16, fontweight='bold', color='green')
    fig.text(0.08, 0.97, "vs.", ha='center', va='center', fontsize=16, fontweight='bold', color='black')
    fig.text(0.12, 0.97, "data in", ha='center', va='center', fontsize=16, fontweight='bold', color='blue')
    fig.text(0.15, 0.97, "vs.", ha='center', va='center', fontsize=16, fontweight='bold', color='black')
    fig.text(0.12, 0.95, "reconst", ha='center', va='center', fontsize=16, fontweight='bold', color='red')
    fig.text(0.08, 0.95, "vs.", ha='center', va='center', fontsize=16, fontweight='bold', color='black')
    fig.text(0.05, 0.95, "mne", ha='center', va='center', fontsize=16, fontweight='bold', color='grey')
    plt.suptitle(f"EEG{fname_tag} - ({batch=}, {idx=}, {sample=}) - NMSE={mse_value:0.5f} - PCC={pcc_value:0.5f} - %dropped={pct_dropout:0.3f}", fontsize=16, fontweight='bold')

    if eeg_signal is not None:
        fig.text(0.8, 0.97, f"NMSE_do={MSE_dropout:0.3f}", ha='center', va='center', fontsize=16, fontweight='bold', color='green')
        fig.text(0.8, 0.95, f"PCC_do={PCC_dropout:0.3f}", ha='center', va='center', fontsize=16, fontweight='bold', color='green')
        fig.text(0.9, 0.97, f"NMSE_~do={MSE_nondrop:0.3f}", ha='center', va='center', fontsize=16, fontweight='bold', color='blue')
        fig.text(0.9, 0.95, f"PCC_~do={PCC_nondrop:0.3f}", ha='center', va='center', fontsize=16, fontweight='bold', color='blue')
    if mne_reconstruction is not None:
        fig.text(0.7, 0.95, f"NMSE_mne={MSE_mne_dropout:0.3f}", ha='center', va='center', fontsize=16, fontweight='bold', color='grey')
        fig.text(0.6, 0.95, f"PCC_mne={PCC_mne_dropout:0.3f}", ha='center', va='center', fontsize=16, fontweight='bold', color='grey')



    plt.savefig(f"{dir_base}/eeg_signal_compare_B{batch}_S{sample}{fname_tag}.png", dpi=300, bbox_inches='tight')
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

    
    data = data.T
    reconst = reconst.T

    num_t, chans = data.shape
    t = np.arange(num_t) #/ fs
    print(f"\tlat: {chans=}, {num_t=}")

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

    # token-space real/pad mask [N] (True=real). When seqlen padding is ON, the
    # packer appends ONE trailing all-zero PAD document (eeg_data.py) whose rows are all
    # pad_mask==0. That doc is NOT a real sample -- unwrapping/scoring/plotting/counting it
    # would corrupt every per-sample metric and add a garbage plot. We detect it below by
    # its all-zero pad_mask slice and skip it. When padding is OFF, pad_mask is all-ones so
    # the skip never triggers and behaviour is byte-identical to before.
    _pad_real = (batch['pad_mask'].reshape(-1).cpu().numpy().astype(bool)
                 if batch.get('pad_mask', None) is not None else None)

    tf = args.data.num_fine_time_pts
    # tc = args.data.seq_len // tf ## THIS ASSUMES TC IS SAME FOR ALL SAMPLES !!

    # Loop through each sample in batch and unwrap the variable-length sequences
    for i,seqlen in enumerate(seq_lens):

        # skip the trailing PAD document (all pad_mask==0 over its slice). Advance
        # the running offset so real-token slicing stays aligned, but emit nothing for it.
        if _pad_real is not None and not _pad_real[seqlen_accum:seqlen_accum+seqlen].any():
            seqlen_accum += seqlen
            continue

        tc = batch['max_tc'][i] ## This allows tc different for each sample
        num_chans = seqlen//tc 

        # SHOULD NOT HAPPEN NOW: This should only happen for the last truncated sample in the batch
        if seqlen != tc*num_chans: #
            print(f"In unwrap_all_the_signals, Sample {i} seqlen {seqlen} != tc {tc} * num_chans {num_chans}")
            seqlen = tc*num_chans
        
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
                                                                                            tc=tc,
                                                                                            use_coarse_time=args.data.use_coarse_time,
            )
            mod_out_sig_unwrapt, mod_out_pos_unwrapt, _, _, _ = invert_reshape_signals(
                                                            sig_reshaped=mod_out_sig, 
                                                            pos_reshaped=mod_out_pos, 
                                                            num_chans=num_chans, 
                                                            tf=tf,
                                                            tc=tc,
                                                            use_coarse_time=args.data.use_coarse_time,
            )
            eeg_sig_unwrapt, _, _, _, _ = invert_reshape_signals(
                                                sig_reshaped=eeg_sig,
                                                num_chans=num_chans, 
                                                tf=tf,
                                                tc=tc,
                                                use_coarse_time=args.data.use_coarse_time,
            )
            lat_data_unwrapt, _, _, _, _ = invert_reshape_signals(
                                                sig_reshaped=lat_data,
                                                num_chans=num_chans, 
                                                tf=tf+3 if args.data.cat_chan_xyz_and_eeg else tf,
                                                tc=tc,
                                                use_coarse_time=args.data.use_coarse_time,
            )
            lat_recon_unwrapt, _, _, _, _ = invert_reshape_signals(
                                                sig_reshaped=lat_recon,
                                                num_chans=num_chans, 
                                                tf=tf+3 if args.data.cat_chan_xyz_and_eeg else tf,
                                                tc=tc,
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
            #
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
        dropped_samps = model_signal_input_unwrapped[samp]==0


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
        MSE_EEG_sig_do = np.abs(model_in_sig[dropped_samps] - model_out_sig[dropped_samps]).mean() # mean square error btwn data and reconst on dropped-out chans
        MSE_EEG_sig_nodo = np.abs(model_in_sig[~dropped_samps] - model_out_sig[~dropped_samps]).mean() # mean square error btwn data and reconst on non-dropped chans
        #


        if dropped_samps.any():
            MAE_EEG_sig_do = compute_mae(model_in_sig[dropped_samps], model_out_sig[dropped_samps])
            NMSE_EEG_sig_do = compute_nmse(model_in_sig[dropped_samps], model_out_sig[dropped_samps])
            SNR_EEG_sig_do = compute_snr(model_in_sig[dropped_samps], model_out_sig[dropped_samps])
            PCC_EEG_sig_do = compute_pcc(model_in_sig[dropped_samps], model_out_sig[dropped_samps])
        else:
            MAE_EEG_sig_do = np.nan
            NMSE_EEG_sig_do = np.nan
            SNR_EEG_sig_do = np.nan
            PCC_EEG_sig_do = np.nan
        #
        if (~dropped_samps).any():
            MAE_EEG_sig_nodo = compute_mae(model_in_sig[~dropped_samps], model_out_sig[~dropped_samps])
            NMSE_EEG_sig_nodo = compute_nmse(model_in_sig[~dropped_samps], model_out_sig[~dropped_samps])
            SNR_EEG_sig_nodo = compute_snr(model_in_sig[~dropped_samps], model_out_sig[~dropped_samps])
            PCC_EEG_sig_nodo = compute_pcc(model_in_sig[~dropped_samps], model_out_sig[~dropped_samps])
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
        do_fft_dropout_stats = False # This made sense to do when we were dropping whole channels. Makes less sense for more general dropout schemes.

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
            dropped_samps = model_signal_input_unwrapped[samp]==0

            fft_sample_data = fft_signal_input_unwrapped[samp]
            fft_sample_recon = fft_signal_output_unwrapped[samp]
            MSEf = np.abs(fft_sample_data - fft_sample_recon).mean() # mean square error btwn data and reconst FFTs   


            if do_fft_dropout_stats:
                MSE_FFT_do = np.abs(fft_sample_data[dropped_samps] - fft_sample_recon[dropped_samps]).mean() # mean square error btwn data and reconst on dropped-out chans
                MSE_FFT_nodo = np.abs(fft_sample_data[~dropped_samps] - fft_sample_recon[~dropped_samps]).mean() # mean square error btwn data and reconst on non-dropped chans
            else:
                MSE_FFT_do = np.nan
                MSE_FFT_nodo = np.nan
            #
            MAE_FFT = compute_mae(fft_sample_data, fft_sample_recon)
            NMSE_FFT = compute_nmse(fft_sample_data, fft_sample_recon)
            SNR_FFT = compute_snr(fft_sample_data, fft_sample_recon)
            PCC_FFT = compute_pcc(fft_sample_data, fft_sample_recon)
            #
            if dropped_samps.any() and do_fft_dropout_stats:
                MAE_FFT_do = compute_mae(fft_sample_data[dropped_samps], fft_sample_recon[dropped_samps])
                NMSE_FFT_do = compute_nmse(fft_sample_data[dropped_samps], fft_sample_recon[dropped_samps])
                SNR_FFT_do = compute_snr(fft_sample_data[dropped_samps], fft_sample_recon[dropped_samps])
                PCC_FFT_do = compute_pcc(fft_sample_data[dropped_samps], fft_sample_recon[dropped_samps])
            else:
                MAE_FFT_do = np.nan
                NMSE_FFT_do = np.nan
                SNR_FFT_do = np.nan
                PCC_FFT_do = np.nan
            #
            if (~dropped_samps).any() and do_fft_dropout_stats:
                MAE_FFT_nodo = compute_mae(fft_sample_data[~dropped_samps], fft_sample_recon[~dropped_samps])
                NMSE_FFT_nodo = compute_nmse(fft_sample_data[~dropped_samps], fft_sample_recon[~dropped_samps])
                SNR_FFT_nodo = compute_snr(fft_sample_data[~dropped_samps], fft_sample_recon[~dropped_samps])
                PCC_FFT_nodo = compute_pcc(fft_sample_data[~dropped_samps], fft_sample_recon[~dropped_samps])
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
        print(f"(mn, std) NMSE for {len(NMSE_samp_EEG_sig)} all      samples of EEG: ({np.nanmean(NMSE_samp_EEG_sig):0.5f}, {np.nanstd(NMSE_samp_EEG_sig):0.5f})")
        print(f"(mn, std) NMSE for {len(NMSE_samp_EEG_sig_do)} drop-out samples of EEG: ({np.nanmean(NMSE_samp_EEG_sig_do):0.5f}, {np.nanstd(NMSE_samp_EEG_sig_do):0.5f})")
        print(f"(mn, std) NMSE for {len(NMSE_samp_EEG_sig_nodo)} non-drop samples of EEG: ({np.nanmean(NMSE_samp_EEG_sig_nodo):0.5f}, {np.nanstd(NMSE_samp_EEG_sig_nodo):0.5f})")
        try:
            print(f"(mn, std) NMSE for {len(NMSE_samp_FFT)} all      samples of FFT: ({np.nanmean(NMSE_samp_FFT):0.5f}, {np.nanstd(NMSE_samp_FFT):0.5f})")
            print(f"(mn, std) NMSE for {len(NMSE_samp_FFT_do)} drop-out samples of FFT: ({np.nanmean(NMSE_samp_FFT_do):0.5f}, {np.nanstd(NMSE_samp_FFT_do):0.5f})")
            print(f"(mn, std) NMSE for {len(NMSE_samp_FFT_nodo)} non-drop samples of FFT: ({np.nanmean(NMSE_samp_FFT_nodo):0.5f}, {np.nanstd(NMSE_samp_FFT_nodo):0.5f})")
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
                            PCC_samp_EEG_sig,
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


            # (1). Plot EEG time course for data and reconstruction on same axis (one ax per channel). One figure per sample.
            if plot_eeg_signal_samples:
                # 1a. Plot with non-dropout signal too.
                plot_compare_eeg_signal(data=model_signal_input_unwrapped[samp],
                                        reconst=model_signal_output_unwrapped[samp],
                                        mse_value=MSE_samp_EEG_sig[samp],
                                        pcc_value=PCC_samp_EEG_sig[samp],
                                        eeg_signal=eeg_signal_unwrapped[samp],
                                        fs=fs,
                                        batch=batch_cntr,
                                        sample=samp,
                                        idx=batch_idx[samp].item(),
                                        fname_tag=""+fname_suptag,
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




# ============================================================================
# 3-way split (see serving_refactor_clode.md in the AY2l repo):
# init_model / init_data_batch / evaluate, called in sequence by main() so the
# compiled model is built + warmed ONCE and can back many data/eval calls.
# Zuna specifics preserved: device (cuda/mps/cpu), HuggingFace weight load,
# wandb forced off, and the V4 .fif reconstructor.
# ============================================================================

@dataclass
class ModelHandle:
    """Warm, compiled model + the distributed context init_model established.
    `model` is None when LOAD_THE_MODEL is False. `device` is a torch.device."""
    model: Optional[Any]
    device: Any
    world_mesh: Any
    dp_rank: int
    dp_degree: int
    model_param_count: int = 0


def _maybe_mark_dynamic_ndoc(t, args):
    """Mark the per-document dim (dim 0 == number of packed documents/users) of a tensor
    dynamic, so every compiled graph taking a per-doc tensor recompiles ONCE for any
    doc-count instead of once per distinct count (fixes 'problem B'). Applies to seq_lens
    (mask builders + encoder/decoder forward) and max_tc (@torch.compile'd EEGProcessor.process).
    Gated by args.dynamic_seq_lens; call EACH iter on the LIVE tensor (mark_dynamic is
    per-tensor, and .to(device) makes a new tensor). Skips size-<=1 dims; never crashes."""
    if not getattr(args, "dynamic_seq_lens", True):
        return
    try:
        if isinstance(t, torch.Tensor) and t.dim() >= 1 and t.shape[0] > 1:
            torch._dynamo.mark_dynamic(t, 0)
    except Exception as e:
        logger.warning(f"[dyn] mark_dynamic(dim 0) skipped: {e}")


def _warmup_model(model, args, device):
    """Compile the model NOW by running one dummy forward at the fixed packed seqlen, so the
    first real request doesn't pay the torch.compile cost. Only meaningful on CUDA (zuna only
    compiles .sample/.encoder there). Inputs are synthesized directly (encoder_input is exactly
    what model.sample takes: [1,S,dim] fp32; tok_idx is int64 [1,S,k], values irrelevant to the
    compile guard). fork_rng + no_grad so the randn here does not perturb per-rank RNG state.
    See the AY2l eeg_eval.py / serving_refactor_clode.md for the full rationale + caveats
    (fixed length requires args.data.pad_packed_seqlen=True; doc-count handled by mark_dynamic)."""
    if not LOAD_THE_MODEL or model is None or getattr(device, "type", None) != "cuda":
        logger.info("[warmup] skipped (model not loaded / not CUDA).")
        return
    try:
        S = int(getattr(args.data, "target_packed_seqlen", args.data.seq_len))
        if not getattr(args.data, "pad_packed_seqlen", False):
            logger.warning("[warmup] pad_packed_seqlen=False: real seqlens vary, so this "
                           "single-length warm-up will NOT prevent recompiles at serve time.")
        tf = int(args.data.num_fine_time_pts)
        dim = tf + (3 if getattr(args.data, "cat_chan_xyz_and_eeg", False) else 0)
        tt = args.model.tok_idx_type
        if tt is None:
            k = None
        elif tt in ("t_coarse", "chan_id", "stack_arange_seqlen"):
            k = 1
        elif tt == "{x,y,z,tc}":
            k = 4
        elif tt == "{x,y,z,tc,ch}":
            k = 5
        else:
            logger.warning(f"[warmup] unknown tok_idx_type={tt!r}; using tok_idx=None.")
            k = None
        sample_steps = int(getattr(args, "diffusion_sample_steps", 50))
        cfg = float(getattr(args, "diffusion_cfg", 1.0))
        ndoc = 8
        base = max(1, S // ndoc)
        lens = [base] * (ndoc - 1) + [S - base * (ndoc - 1)]
        lens = [l for l in lens if l > 0]

        t0 = timer()
        with torch.no_grad(), torch.random.fork_rng(devices=[device]):
            torch.manual_seed(0)
            encoder_input = torch.randn(1, S, dim, device=device, dtype=torch.float32)
            encoder_input[:, ::7, :] = 0.0  # exercise the dropped/padded-token path
            seq_lens = torch.tensor(lens, device=device, dtype=torch.long)
            _maybe_mark_dynamic_ndoc(seq_lens, args)
            tok_idx = None if k is None else torch.zeros(1, S, k, dtype=torch.long)  # CPU long, like real
            logger.info(f"[warmup] compiling: S={S} dim={dim} k={k} "
                        f"sample_steps={sample_steps} cfg={cfg} docs={len(lens)}")
            z, _ = model.sample(encoder_input=encoder_input, seq_lens=seq_lens,
                                tok_idx=tok_idx, cfg=cfg, sample_steps=sample_steps)
            model.encoder(token_values=encoder_input, seq_lens=seq_lens, tok_idx=tok_idx)
            model.encoder(token_values=z, seq_lens=seq_lens, tok_idx=tok_idx)
        torch.cuda.synchronize()
        logger.info(f"[warmup] done in {timer() - t0:.1f}s -- model is warm.")
    except Exception as e:
        logger.warning(f"[warmup] skipped due to error: {type(e).__name__}: {e}")


def init_model(args: TrainArgs, device: Optional[torch.device] = None) -> ModelHandle:
    """Phase 1/3: device select + distributed/env setup + build + (HF or local) weight load
    + torch.compile of model.sample/.encoder + warm-up. Runs ONCE; returns a warm ModelHandle.
    MetricLogger / ExitStack live in evaluate(), not here."""
    model = None
    model_param_count = 0
    if torch.cuda.is_available():
        device = device or torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = device or torch.device("mps")
        os.environ['TORCH_COMPILE_DISABLE'] = "1"
        os.environ['TORCHDYNAMO_DISABLE'] = "1"
    else:
        device = device or torch.device("cpu")
        os.environ['TORCH_COMPILE_DISABLE'] = "1"
        os.environ['TORCHDYNAMO_DISABLE'] = "1"

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

    # Initializing Model in meta device allows us to initialize models much bigger than 1 gpu's memory
        
    if LOAD_THE_MODEL:
        Load_from_HF = True
        if Load_from_HF:
            # ===== Load model + weights from HuggingFace (Zyphra/ZUNA) =====
            # Toggle the `if True` to `if False` to fall through to the local
            # checkpoint / ema.pt loader in the else branch below.
            # device selected above (cuda / mps / cpu)

            def load_model_args_from_hf(repo_id: str, config_filename: str = "config.json") -> DecoderTransformerArgs:
                config_path = hf_hub_download(repo_id=repo_id, filename=config_filename)
                with open(config_path, "r") as f:
                    cfig = json.load(f)
                return dataclass_from_dict(DecoderTransformerArgs, cfig["model"])

            REPO_ID = "Zyphra/ZUNA1.1"
            WEIGHTS = "model-00001-of-00001.safetensors"
            CONFIG  = "config.json"

            model_args = load_model_args_from_hf(REPO_ID, CONFIG)
            weights_path = hf_hub_download(repo_id=REPO_ID, filename=WEIGHTS, token=False)
            sd_st_raw = safe_load(weights_path, device="cpu")

            # Normalize: strip leading "model." if present
            sd_st = {k.removeprefix("model."): v for k, v in sd_st_raw.items()}

            model = EncoderDecoder(model_args).to(device)
            sd_st_on_dev = {k: v.to(device) for k, v in sd_st.items()}
            model.load_state_dict(sd_st_on_dev, strict=True)
            model.eval()

            if device.type == "cuda":
                model.sample = torch.compile(model.sample)
                model.encoder = torch.compile(model.encoder)
        else:
            with torch.device("meta"):
                model = EncoderDecoder(args.model)

            logger.info("Model is built !")

            model_param_count = get_num_params(model)

            if device.type == "cuda":
                model.sample = torch.compile(model.sample)  # compiled despite graph breaks from the sampling loop in .sample
                model.encoder = torch.compile(model.encoder)

            # Once we shard the model on different gpus we can actually initialize the model
            # First we create empty tensors of the correct shapes
            model = model.to_empty(device=device) # Use local device, not cuda:0
            # Then we init the model. Please make sure this function initializes *ALL* parameters
            # and buffers, otherwise you will have random values in the unitialized tensors
            # which will silently fail (give nan gradients for example)

            if args.checkpoint.init_ckpt_path:
                with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                    torch.manual_seed(args.model.seed)
                    model.init_weights()
                check_model_value_range(model, range=10.0, std=1.0)
                logger.info(f"!!!! Loading initial model from {args.checkpoint.init_ckpt_path} !!!! \n\n")
                load_from_checkpoint(args.checkpoint.init_ckpt_path, model, model_key="model") # Put model_key="" if its directly the model checkpoint
                logger.info("!!!!!!!!!!! Model loaded from checkpoint completed !!!!!!!!!!!")
                check_model_value_range(model, range=10.0, std=1.0)
            else:
                with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                    torch.manual_seed(args.model.seed)
                    model.init_weights()
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
                logger.info("Running on CPU/MPS")

            # Model weights are fully loaded above via load_from_checkpoint(init_ckpt_path).
            # The training-resume path (build_optimizer + TrainState + CheckpointManager, which
            # needs args.checkpoint.path) is not needed for inference and has been removed. 
            if getattr(args.optim, "use_ema", False):
                from apps.AY2latent_bci.ema import EMA
                _ema = EMA(model)                                  # scaffold; shadow = current weights
                if _ema.maybe_load(args.checkpoint.init_ckpt_path):  # ema.pt sits IN the step dir
                    _ema.copy_to(model)                            # overwrite weights with EMA (throwaway eval -> safe)
                    logger.info(f"[EMA] applied {args.checkpoint.init_ckpt_path}/ema.pt to eval model")
                else:
                    logger.warning(f"[EMA] use_ema set but no ema.pt in {args.checkpoint.init_ckpt_path}; using raw weights")


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

    if LOAD_THE_MODEL:
        for p in model.parameters():
            p.requires_grad = False # True (False for eval, True for training)

    _warmup_model(model, args, device)

    return ModelHandle(model=model, device=device,
                       world_mesh=world_mesh, dp_rank=dp_rank, dp_degree=dp_degree,
                       model_param_count=model_param_count)


def init_data_batch(args: TrainArgs, handle: ModelHandle):
    """Phase 2/3: build the (offline) data source, the V4 .fif reconstructor (if enabled),
    and the EEGProcessor. Returns (data_loader, batch_iterator, data_processor, v4_reconstructor).
    Offline only -- at serve time call this externally on a pool of submitted requests treated
    like an offline dataset. Per-batch process()/.to(device)/tok_idx stay in evaluate()'s loop."""
    dp_rank = handle.dp_rank
    device = handle.device
    print("Entering create dataloader on rank", dp_rank)
    data_loader = create_dataloader_v2(args.data, args.seed, dp_rank)
    print("Finishing create dataloader on rank", dp_rank)


    # V4 .fif save-out: buffer per-segment model output, stitch to continuous .fif.
    v4_reconstructor = None
    if getattr(args.data, "use_v4", False) and getattr(args.data, "v4_recon_save_fif", False):
        _recon_out = getattr(args.data, "v4_recon_out_dir", None) or args.dump_dir
        v4_reconstructor = FifReconstructor(
            output_dir=_recon_out,
            raw_info_registry=data_loader.dataset.raw_info_registry,
            fill_only_masked=args.data.v4_recon_fill_only_masked,
            num_fine_time_pts=args.data.num_fine_time_pts,
            data_norm=getattr(args.data, "data_norm", 1.0),
            unmasked_from_original=getattr(args.data, "v4_recon_unmasked_from_original", False),
            seam_correct=getattr(args.data, "v4_recon_seam_correct", True),
            annotate_infill=getattr(args.data, "v4_recon_annotate_infill", True),
        )
        print(f"[v4 recon] enabled -> {_recon_out} (full_reconstruction/ + hybrid/)")

    epoch = 0 # if using nonlocal epoch
    def make_batch_iterator(dataloader, data_args):  # Use with IterableDataset.
        """
        Moving sequence packing into Dataset/Dataloader/Collator. Too slow when done here.
        """
        nonlocal epoch
        print("Creating batch iterator of dataloader with length", len(dataloader), "and dataset of length", len(dataloader.dataset))

        eeg_sig_norm = data_args.data_norm # normalization factor for eeg signal.
        eeg_sig_clip = data_args.data_clip # clipping factor for eeg signal.

        while True:
            epoch += 1
            logger.info(f"Starting epoch: {epoch}")
            for idx,batch in enumerate(dataloader):


                eeg_signal = batch['eeg_signal']

                eeg_signal = eeg_signal/eeg_sig_norm # Divide by eeg_sig_norm to normalize the data and change its STD.

                if eeg_sig_clip is not None:
                    print(f"Clipping input at +/-{eeg_sig_clip}")
                    eeg_signal = eeg_signal.clamp(min=-eeg_sig_clip, max=eeg_sig_clip) # 

                yielded = {
                    'eeg_signal': eeg_signal, # pass out the clipped and normalized eeg signal.
                    'chan_pos': batch['chan_pos'],
                    'chan_pos_discrete': batch['chan_pos_discrete'],
                    'chan_id': batch['chan_id'],
                    't_coarse': batch['t_coarse'],
                    'token_dropout': batch['token_dropout'],
                    'seq_lens': batch['seq_lens'],
                    'max_tc': batch['max_tc'],
                    'pad_mask': batch['pad_mask'],
                    'idx': batch['ids'],
                    'dataset_id': batch['dataset_id'],
                }
                # Pass V4 reconstruction metadata through (no-op for V2/V3/B2).
                for _k in ('v4_seg_mean', 'v4_seg_std', 'v4_avg_ref_offset',
                           'v4_fif_path', 'v4_seg_start', 'v4_seg_end',
                           'v4_channel_names', 'v4_sfreq', 'v4_raw_info',
                           'v4_unfiltered_volts', 'v4_step_times'):
                    if _k in batch:
                        yielded[_k] = batch[_k]
                yield yielded

            print("Finished epoch", epoch)
            # V4 is single-pass inference — stop after one full walk of the .fif files.
            if getattr(data_args, "use_v4", False):
                print("[v4] one full pass through all .fif files complete — stopping iterator")
                return

    batch_iterator = make_batch_iterator(data_loader, args.data)
    print("Entering create batch iterator on rank", dp_rank)

    # fixed-eval: plot/score the SAME frozen samples that back the training
    # curve. Load the identical on-disk pool the training run built (keyed by
    # data_dir/seed/N), take the first `plot_num_batches`, and seed the sampler
    # per GLOBAL pool index so each reconstruction is byte-identical to training.
    # See eval_harness.py. NOTE: for coherence the training run (or a
    # pre-build) should create the pool file first; if it is missing here, this
    # single process builds it from its own (world_size=1) data stream.
    if getattr(args.data, "fixed_eval", False):
        from apps.AY2latent_bci.eval_harness import (
            build_or_load_fixed_eval_set, sample_noise_seed, fixed_eval_cache_path,
        )
        _pool = build_or_load_fixed_eval_set(args.data, args.seed, get_is_master())
        _subset = _pool[: args.data.plot_num_batches]
        logger.info(f"[fixed-eval] eeg_eval: plotting/scoring {len(_subset)}/{len(_pool)} "
                    f"frozen samples from {fixed_eval_cache_path(args.data, args.seed)}")

        def _frozen_batch_iter(subset, base_seed):
            for gi, b in enumerate(subset):
                torch.manual_seed(sample_noise_seed(base_seed, gi))  # process()+sample() noise
                yield {k: v for k, v in b.items()}   # shallow copy; loop's .pop() won't mutate the cache
        batch_iterator = _frozen_batch_iter(_subset, args.data.eval_noise_seed)

    data_processor = EEGProcessor(args.data).to(device)

    return data_loader, batch_iterator, data_processor, v4_reconstructor


def evaluate(args: TrainArgs, handle: ModelHandle, src):
    """Phase 3/3: the eval loop over prepared data, using the already-warm model. `handle` from
    init_model(); `src` == (data_loader, batch_iterator, data_processor, v4_reconstructor) from
    init_data_batch(). Metrics + plotting + V4 recon still live here (see serving_refactor_clode.md
    'Follow-ups' for the lean infer_step extraction)."""
    fs = args.data.sample_rate
    num_t = args.data.seq_len

    model = handle.model
    device = handle.device
    dp_rank = handle.dp_rank
    world_mesh = handle.world_mesh
    data_loader, batch_iterator, data_processor, v4_reconstructor = src

    with ExitStack() as context_stack:
        # Inference codebase: never touch Weights & Biases (see original note).
        if getattr(args, "logging", None) is not None:
            args.logging.wandb = None
        metric_logger = context_stack.enter_context(
            MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args)
        )
        torch_profiler = None
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #
        # (2). Here, run EEG data through autoencoder model and compare model input with model output
        #       TO DO: Compare dec_out to batch['encoder_input'] or eeg_signal.
        #

        plot_eeg_signal_samples = args.plot_eeg_signal_samples      # Plot raw eeg for data and model reconstruction for single samples
        plot_eeg_position_samples = False #True   # Scatter eeg channel position GT vs reconstruction for single samples
        plot_fft_samples = False #True             # Plot fft of eeg for data and model reconstruction for single samples
        plot_latent_samples = False #True
        compute_encoder_consistency = True
        # Recon-only v4 runs skip the per-sample metrics + eval figures (the *_vs_* plots
        # under the checkpoint dir): we only want the reconstructed .fif. Much faster.
        compute_reconstruction_metrics_stats_across_dataset = (v4_reconstructor is None)

        sample_steps = args.diffusion_sample_steps    # for diffusion process in .sample
        cfg = args.diffusion_cfg            # for diffusion process in .sample (1.0 = no cfg)


        dir_base = args.inference_figures_dir 
        if args.plot_eeg_signal_samples:
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



        # plot/score exactly this many samples (subset of the frozen pool in
        # fixed_eval mode). Was hardcoded to 5; now driven by config plot_num_batches.
        # V4 single-pass: process ALL segments (natural stop = StopIteration); lift the cap.
        num_batches = 10**9 if getattr(args.data, "use_v4", False) else getattr(args.data, "plot_num_batches", 5)
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
            #
            num_samples_list = []
            num_channels_list = []
            pct_dropout_list = []



        while True:
            try:
                batch = next(batch_iterator)
            except StopIteration:
                print(f"[v4] batch iterator exhausted after {batch_cntr} batches")
                break
            batch_cntr += 1


            eeg_signal = batch['eeg_signal']

            batch_idx = batch.pop('idx', None)
            batch_dataset_id = batch.pop('dataset_id', None)   # NOTE: pop takes them out of batch. if left in, breaks things below and not training on these.
            

            # Pop V4 reconstruction metadata before process(**batch)/.cuda(); re-attach after.
            v4_meta_keys = ('v4_seg_mean', 'v4_seg_std', 'v4_avg_ref_offset',
                            'v4_fif_path', 'v4_seg_start', 'v4_seg_end',
                            'v4_channel_names', 'v4_sfreq', 'v4_raw_info', 'v4_unfiltered_volts')
            v4_meta = {k: batch.pop(k) for k in v4_meta_keys if k in batch}
            batch.pop('v4_step_times', None)
            v4_token_dropout = batch.get('token_dropout', None)
            # process() is @torch.compile'd and takes per-doc tensors
            # (seq_lens, max_tc); mark their doc-count dim dynamic BEFORE the call.
            _maybe_mark_dynamic_ndoc(batch['seq_lens'], args)
            _maybe_mark_dynamic_ndoc(batch['max_tc'], args)
            with torch.no_grad():
                batch = data_processor.process(**batch)                             #  > option 3.

            batch = {k: v.to(device, non_blocking=(device.type == "cuda")) for k, v in batch.items()}
            # .to(device) made fresh tensors -> re-mark the cuda seq_lens
            # that enters the model (encoder/decoder/mask graphs).
            _maybe_mark_dynamic_ndoc(batch['seq_lens'], args)
            if v4_reconstructor is not None:
                batch.update(v4_meta)   # re-attach non-tensor recon metadata
                if v4_token_dropout is not None:
                    batch['token_dropout'] = v4_token_dropout.to(device, non_blocking=(device.type == "cuda"))

            tf = args.data.num_fine_time_pts
            tc = args.data.seq_len // tf # This would assume tc is same for all samples, but is overwritten below by max_tc for each sample.

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
            elif args.model.tok_idx_type == "{x,y,z,tc}" and args.model.rope_dim==4: # 4D-RoPE on {x,y,z,tc} w/ no APE on chan_id
                chan_pos_discrete = batch['chan_pos_discrete'].cpu().unsqueeze(0)      # [1, seqlen, 3]
                t_coarse = batch['t_coarse'].cpu().unsqueeze(0)                        # [1, seqlen, 1]
                tok_idx = torch.cat((chan_pos_discrete,t_coarse), dim=2)               # [1, seqlen, 4]
            elif args.model.tok_idx_type == "{x,y,z,tc,ch}" and args.model.rope_dim==4 and args.model.ape_dim==1: # 4D-RoPE on {x,y,z,tc} + 1D-APE on chan_id
                chan_pos_discrete = batch['chan_pos_discrete'].cpu().unsqueeze(0)      # [1, seqlen, 3]
                t_coarse = batch['t_coarse'].cpu().unsqueeze(0)                        # [1, seqlen, 1]
                chan_id = batch['chan_id'].cpu().unsqueeze(0)                          # [1, seqlen, 1]
                tok_idx = torch.cat((chan_pos_discrete,t_coarse,chan_id), dim=2)       # [1, seqlen, 5]
            else:
                print(f"Dont understand {args.model.tok_idx_type=} and {args.model.rope_dim}")
                die

            # If zero_spatial is True, zero out the spatial dimensions of tok_idx to mimic data with no channel {x,y,z} position information.
            if args.model.zero_spatial and tok_idx is not None:
                tok_idx[:, :, :3] = 0



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
                    latent_data, _, _ = model.encoder(
                                            token_values=batch['encoder_input'].unsqueeze(0), 
                                            seq_lens=batch['seq_lens'],
                                            tok_idx=tok_idx,
                    )
                    latent_recon, _, _ = model.encoder(
                                            token_values=z, #z_masked, 
                                            seq_lens=batch['seq_lens'],
                                            tok_idx=tok_idx,
                    )
            else:
                latent_data, latent_recon = None, None




            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     


            signals_to_plot = []
            # signals_to_plot = inference_at_step # [ inference_at_step[-2] ] # UNCOMMENT IF YOU WANT TO PLOT THE INTERMEDIATE STEPS OF THE DIFFUSION PROCESS
                                                                              # NOTE: If computing reconstruction-based metrics, we need to plot only the final sample from the diffusion process.

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


                
                # Buffer per-segment reconstructions for .fif save-out.
                if v4_reconstructor is not None:
                    v4_reconstructor.add_batch(
                        batch=batch,
                        model_signal_input_unwrapped=model_signal_input_unwrapped,
                        model_signal_output_unwrapped=model_signal_output_unwrapped,
                        channel_id_unwrapped=channel_id_unwrapped,
                        t_coarse_unwrapped=t_coarse_unwrapped,
                    )
                    # Recon-only fast path: the .fif is buffered above, so skip the per-sample
                    # MNE interpolation, reconstruction metrics and eval figures below.
                    continue

                # Prepare channel positions for MNE - now, tc can be different for each sample.
                chan_pos_list = []
                for i in range(len(model_signal_input_unwrapped)):
                    tc = batch['max_tc'][i].item()
                    chan_pos_list.append(model_position_input_unwrapped[i].reshape(-1, tc, 3)[:, 0, :])


                # Apply MNE interpolation to dropped-out channels
                mne_interpolated_signals = interpolate_signals_with_mne(
                    signals=model_signal_input_unwrapped,
                    channel_positions=chan_pos_list,
                    sampling_rate=fs,
                    mark_zero_variance=True
                )

                # Compute FFT of signal input into model and signal output from model.
                fft_signal_input_unwrapped, freqs = compute_sig_FFT(eeg_signal_unwrapped, fs) # non-dropped-out signal.
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
                                       NMSE_samp_EEG_sig, 
                                       PCC_samp_EEG_sig,
                                       #
                                       model_position_input_unwrapped, 
                                       model_position_output_unwrapped, 
                                       NMSE_samp_EEG_pos, 
                                       #
                                       fft_signal_input_unwrapped, 
                                       fft_signal_output_unwrapped,
                                       MSE_samp_FFT,
                                       MSE_samp_FFT_do,
                                       MSE_samp_FFT_nodo,
                                       #
                                       latent_data_unwrapped,
                                       latent_recon_unwrapped,
                                       NMSE_samp_latent, 
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
                #
                num_samples_list.extend([samp.shape[1] for samp in model_signal_input_unwrapped])
                num_channels_list.extend([samp.shape[0] for samp in model_signal_input_unwrapped])
                pct_dropout_list.extend([(samp==0).sum()/samp.size for samp in model_signal_input_unwrapped])





                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

                good_do_idx = ~np.isnan(NMSE_samp_EEG_sig_do_list)
                good_nodo_idx = ~np.isnan(NMSE_samp_EEG_sig_nodo_list)

                # Plot num_samples_list vs NMSE_samp_EEG_sig_do_list
                plt.figure(figsize=(10, 10))
                plt.scatter(np.array(num_samples_list)[good_do_idx].astype(float), np.array(NMSE_samp_EEG_sig_do_list)[good_do_idx].astype(float), marker='o', color='g', alpha=0.3)
                plt.scatter(np.array(num_samples_list)[good_nodo_idx].astype(float), np.array(NMSE_samp_EEG_sig_nodo_list)[good_nodo_idx].astype(float), marker='o', color='b', alpha=0.3)
                plt.yscale('log')
                plt.ylim(1e-4, 10)
                plt.grid(True)
                plt.xlabel('Number of samples')
                plt.ylabel('NMSE')
                plt.title('NMSE vs Number of samples')
                plt.savefig(f'{dir_base}/NMSE_vs_num_samples.png')
                plt.close()

                # Plot num_channels_list vs NMSE_samp_EEG_sig_do_list
                plt.figure(figsize=(10, 10))
                plt.scatter(np.array(num_channels_list)[good_do_idx].astype(float), np.array(NMSE_samp_EEG_sig_do_list)[good_do_idx].astype(float), marker='o', color='g', alpha=0.3)
                plt.scatter(np.array(num_channels_list)[good_nodo_idx].astype(float), np.array(NMSE_samp_EEG_sig_nodo_list)[good_nodo_idx].astype(float), marker='o', color='b', alpha=0.3)
                plt.yscale('log')
                plt.ylim(1e-4, 10)
                plt.grid(True)
                plt.xlabel('Number of channels')
                plt.ylabel('NMSE')
                plt.title('NMSE vs Number of channels')
                plt.savefig(f'{dir_base}/NMSE_vs_num_channels.png')
                plt.close()

                # Plot pct_dropout_list vs NMSE_samp_EEG_sig_do_list
                plt.figure(figsize=(10, 10))
                plt.scatter(np.array(pct_dropout_list)[good_do_idx].astype(float), np.array(NMSE_samp_EEG_sig_do_list)[good_do_idx].astype(float), marker='o', color='g', alpha=0.3)
                plt.scatter(np.array(pct_dropout_list)[good_nodo_idx].astype(float), np.array(NMSE_samp_EEG_sig_nodo_list)[good_nodo_idx].astype(float), marker='o', color='b', alpha=0.3)
                plt.yscale('log')
                plt.ylim(1e-4, 10)
                plt.xlim(0.0, 1.0)
                plt.grid(True)
                plt.xlabel('Percentage of dropout')
                plt.ylabel('NMSE')
                plt.title('NMSE vs Percentage of dropout')
                plt.savefig(f'{dir_base}/NMSE_vs_pct_dropout.png')
                plt.close()


                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

                good_do_idx = ~np.isnan(PCC_samp_EEG_sig_do_list)
                good_nodo_idx = ~np.isnan(PCC_samp_EEG_sig_nodo_list)

                # Plot num_samples_list vs PCC_samp_EEG_sig_do_list
                plt.figure(figsize=(10, 10))
                plt.scatter(np.array(num_samples_list)[good_do_idx].astype(float), np.array(PCC_samp_EEG_sig_do_list)[good_do_idx].astype(float), marker='o', color='g', alpha=0.3)
                plt.scatter(np.array(num_samples_list)[good_nodo_idx].astype(float), np.array(PCC_samp_EEG_sig_nodo_list)[good_nodo_idx].astype(float), marker='o', color='b', alpha=0.3)
                plt.ylim(-1, 1)
                plt.grid(True)
                plt.xlabel('Number of samples')
                plt.ylabel('PCC')
                plt.title('PCC vs Number of samples')
                plt.savefig(f'{dir_base}/PCC_vs_num_samples.png')
                plt.close()

                # Plot num_channels_list vs PCC_samp_EEG_sig_do_list
                plt.figure(figsize=(10, 10))
                plt.scatter(np.array(num_channels_list)[good_do_idx].astype(float), np.array(PCC_samp_EEG_sig_do_list)[good_do_idx].astype(float), marker='o', color='g', alpha=0.3)
                plt.scatter(np.array(num_channels_list)[good_nodo_idx].astype(float), np.array(PCC_samp_EEG_sig_nodo_list)[good_nodo_idx].astype(float), marker='o', color='b', alpha=0.3)
                plt.ylim(-1, 1)
                plt.grid(True)
                plt.xlabel('Number of channels')
                plt.ylabel('PCC')
                plt.title('PCC vs Number of channels')
                plt.savefig(f'{dir_base}/PCC_vs_num_channels.png')
                plt.close()

                # Plot pct_dropout_list vs PCC_samp_EEG_sig_do_list
                plt.figure(figsize=(10, 10))
                plt.scatter(np.array(pct_dropout_list)[good_do_idx].astype(float), np.array(PCC_samp_EEG_sig_do_list)[good_do_idx].astype(float), marker='o', color='g', alpha=0.3)
                plt.scatter(np.array(pct_dropout_list)[good_nodo_idx].astype(float), np.array(PCC_samp_EEG_sig_nodo_list)[good_nodo_idx].astype(float), marker='o', color='b', alpha=0.3)
                plt.ylim(-1, 1)
                plt.xlim(0.0, 1.0)
                plt.grid(True)
                plt.xlabel('Percentage of dropout')
                plt.ylabel('PCC')
                plt.title('PCC vs Percentage of dropout')
                plt.savefig(f'{dir_base}/PCC_vs_pct_dropout.png')
                plt.close()


                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



            # Here if you want to only do a certain number of batches (like for making a couple plots))
            if batch_cntr >= num_batches:
                break

            # # Here if you want to only do a certain number of epochs (like for computng eval metric stats)
            # if epoch > 1:
            #     break

        # V4: stitch buffered segments into continuous .fif (full_reconstruction/ + hybrid/).
        if v4_reconstructor is not None:
            v4_reconstructor.save_all()
            return   # recon-only run: per-sample metrics were skipped, nothing to print

        ## Display Stats of reconstruction-based metrics across batches of data
        try:
            print(f"\n\n{len(MAE_samp_EEG_mne_do_list)} samples from {data_loader.dataset.key_prefix} with token dropout rate {args.data.token_dropout_prob}") # backblaze path in EEGDataset_b2
        except:
            try:
                print(f"\n\n{len(MAE_samp_EEG_mne_do_list)} samples from {data_loader.dataset.memmap_paths[0].parts[5]} with token dropout rate {args.data.token_dropout_prob}") # local path in EEGDataset_v2
            except:
                print(f"\n\nUsing EEGDataset_v3")

        print("\nMAE:")
        print(f"\tZUNA recon: (mean +/- std) ({np.nanmean(np.array(MAE_samp_EEG_sig_do_list)):.4f} +/- {np.nanstd(np.array(MAE_samp_EEG_sig_do_list)):.4f})")
        print(f"\tmne interp: (mean +/- std) ({np.nanmean(np.array(MAE_samp_EEG_mne_do_list)):.4f} +/- {np.nanstd(np.array(MAE_samp_EEG_mne_do_list)):.4f})")
        print("NMSE:")
        print(f"\tZUNA recon: (mean +/- std) ({np.nanmean(np.array(NMSE_samp_EEG_sig_do_list)):.4f} +/- {np.nanstd(np.array(NMSE_samp_EEG_sig_do_list)):.4f})")
        print(f"\tmne  interp: (mean +/- std) ({np.nanmean(np.array(NMSE_samp_EEG_mne_do_list)):.4f} +/- {np.nanstd(np.array(NMSE_samp_EEG_mne_do_list)):.4f})")
        print("SNR:")
        print(f"\tZUNA recon: (mean +/- std) ({np.nanmean(np.array(SNR_samp_EEG_sig_do_list)):.4f} +/- {np.nanstd(np.array(SNR_samp_EEG_sig_do_list)):.4f})")
        print(f"\tmne interp: (mean +/- std) ({np.nanmean(np.array(SNR_samp_EEG_mne_do_list)):.4f} +/- {np.nanstd(np.array(SNR_samp_EEG_mne_do_list)):.4f})")
        print("PCC:") 
        print(f"\tZUNA recon: (mean +/- std) ({np.nanmean(np.array(PCC_samp_EEG_sig_do_list)):.4f} +/- {np.nanstd(np.array(PCC_samp_EEG_sig_do_list)):.4f})")
        print(f"\tmne interp: (mean +/- std) ({np.nanmean(np.array(PCC_samp_EEG_mne_do_list)):.4f} +/- {np.nanstd(np.array(PCC_samp_EEG_mne_do_list)):.4f})")
        print(f"\n\n")


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

    # 3-way split: build+warm the model once, prepare data, then run the loop.
    # For serving, init_model() can be called once and reused across many
    # init_data_batch()/evaluate() calls on different pools of submitted requests.
    handle = init_model(cfig)
    src = init_data_batch(cfig, handle)
    evaluate(cfig, handle, src)


if __name__ == "__main__":
    main()
