import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
# import zarr
import numpy as np
import math
import json  #jm
from dataclasses import dataclass, field
from typing import Union, List, Optional
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import random
import os
from dotenv import load_dotenv
import time
from pathlib import Path

import matplotlib.pyplot as plt

import boto3
import tempfile
import logging
import fnmatch

logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('s3transfer').setLevel(logging.WARNING)



def chop_and_reshape_signals(eeg_signal, chan_pos=None, chan_pos_discrete=None, tf=128, use_coarse_time="B"):
    """
    This reshapes an eeg_signal that is Size(ch,tpts) into something that either

        (1a). interleaves channels and coarse time along one dimension keeping coarse-time together if use_coarse_time=="A"
           [ch1,tc1: ch2,tc1: ... chN,tc1: --->
            ch1,tc2: ch2,tc2: ... chN,tc2: ---> 
            ch1,tcK: ch2,tcK: ... chN,tcK]
    or
        (1b). interleaves channels and coarse time along one dimension keeping channels together if use_coarse_time=="B"
           [ch1,tc1: ch1,tc2: ... ch1,tck: --->
            ch2,tc1: ch2,tc2: ... ch2,tck: ---> 
            chN,tc1: chN,tc2: ... chN,tck]
    or
        (1c). grabs just first coarse time chunk (tc=1) for all channels if use_coarse_time=="C"
           [ch1,tc1: ch2,tc1: ... chN,tc1]  
    or
        (1d). similar to B, but splits each channel into its own sample if use_coarse_time=="D"
           [[ch1,tc1: ch1,tc2: ... ch1,tck]
            [ch2,tc1: ch2,tc2: ... ch2,tck] 
            [chN,tc1: chN,tc2: ... chN,tck]]          

    and 
        (2). has the fine time sequence along the other dimension

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    Test it out with this example:
        tf = 16
        tc = 10
        num_chans = 21
        #
        mc = torch.zeros(num_chans,tf*tc)   # Labeled Channels
        mt = torch.zeros(num_chans,tf*tc)   # Labeled time_pts
        cp = torch.zeros(num_chans,3)       # Labeled Channel {x,y,z}-positions
        #
        for i in range(num_chans):
            cp[i,0] = i + 0.0       # label for x
            cp[i,1] = i + 0.1       # label for y
            cp[i,2] = i + 0.2       # label for z
            for j in range(tf*tc):
                mc[i,j] = i
                mt[i,j] = j
        #
        nc, cpr, cpdr, cir, tcr, sql = chop_and_reshape_signals(eeg_signal=mc, chan_pos=cp, chan_pos_discrete=cp, tf=tf, use_coarse_time="B"|"A"|"C")
        nt, cpr, cpdr, cir, tcr, sql = chop_and_reshape_signals(eeg_signal=mt, chan_pos=cp, chan_pos_discrete=cp, tf=tf, use_coarse_time="B"|"A"|"C")

        # inspect nc, nt, cpr, cpdr, cir, tcr, sql
    
    Expected results:
        sql = num_chans*tc
        nc.shape = nt.shape = (sql,num_chans)
        cpr.shape = (sql,3)
        cpdr.shape = (sql,3)
        cir.shape = tcr.shape = (sql,1)

    """
    num_chans, num_tpts = eeg_signal.shape

    if use_coarse_time=="C":
        tc = 1
    else:
        # coarse_time=="A"|"B"|"D"
        assert num_tpts%tf==0, f"{num_tpts=} is not divisible by tf={tf}. {num_chans=}"
        tc = num_tpts//tf


    if use_coarse_time=="A":
        # Keep same coarse-time values together in reshaping.
        seqlen = num_chans*tc
        eeg_reshaped = eeg_signal.reshape(num_chans, tc, tf).transpose(0,1).reshape(seqlen,tf)
        chan_pos_reshaped = chan_pos.repeat((tc,1)) if chan_pos is not None else None
        chan_pos_discrete_reshaped = chan_pos_discrete.repeat((tc,1)) if chan_pos_discrete is not None else None
        chan_id_reshaped = torch.arange(num_chans).unsqueeze(-1).repeat((tc,1))
        tc_reshaped = torch.arange(tc).repeat((num_chans,1)).T.reshape(seqlen,1)

    elif use_coarse_time=="B" or use_coarse_time=="D":
        # THIS IS DEFAULT: Keep same channels together in reshaping
        seqlen = num_chans*tc
        eeg_reshaped = eeg_signal.reshape(num_chans, tc, tf).reshape(seqlen,tf)
        chan_pos_reshaped = chan_pos.repeat_interleave(repeats=tc,dim=0) if chan_pos is not None else None
        chan_pos_discrete_reshaped = chan_pos_discrete.repeat_interleave(repeats=tc,dim=0) if chan_pos_discrete is not None else None
        chan_id_reshaped = torch.arange(num_chans).unsqueeze(-1).repeat_interleave(repeats=tc,dim=0) 
        tc_reshaped = torch.arange(tc).repeat((num_chans,1)).reshape(seqlen,1)

    elif use_coarse_time=="C":
        # just grab the first tf time points
        seqlen = num_chans
        eeg_reshaped = eeg_signal[:, :tf]  
        chan_pos_reshaped = chan_pos
        chan_pos_discrete_reshaped = chan_pos_discrete
        tc_reshaped = torch.zeros(num_chans,1)
        chan_id_reshaped = torch.arange(num_chans).unsqueeze(-1)

    else:
        print(f"Not implemented error: {use_coarse_time=} and it needs to be A, B, C or D.")
        die

    if use_coarse_time=="D":
        # Keep same channels together in reshaping then split each channel into its own sample.
        # NOT SURE I CAN INVERT THIS IN INVERT_RESHAPE_SIGNALS.

        # pack each channel separately into list
        indx = list(range(0,tc*num_chans,tc))
        eegr = []
        cpr = []
        cpdr = []
        tcr = []
        cir = []
        sql = []
        for i in indx:
            st, nd = i, i+tc  
            eegr.append( eeg_reshaped[st:nd,:] )
            cpr.append( chan_pos_reshaped[st:nd,:]  )
            cpdr.append( chan_pos_discrete_reshaped[st:nd,:]  )
            tcr.append( tc_reshaped[st:nd,:] )
            cir.append( chan_id_reshaped[st:nd,:] )
            sql.append(tc)
        #
        eeg_reshaped = eegr
        chan_pos_reshaped = cpr
        chan_pos_discrete_reshaped = cpdr
        tc_reshaped = tcr
        chan_id_reshaped = cir
        seqlen = sql


    ## For "A" and "B", ...  ("C" and "D" are different)
    # eeg_reshaped.shape = [num_chans*tc, tf]
    # chan_pos_reshaped.shape = [num_chans*tc, 3]
    # tc_reshaped.shape = [num_chans*tc, 3] 
    # num_chans*tc = int
    return eeg_reshaped, chan_pos_reshaped, chan_pos_discrete_reshaped, chan_id_reshaped, tc_reshaped, seqlen, num_chans




def invert_reshape_signals(sig_reshaped, pos_reshaped=None, pos_discrete_reshaped=None, id_reshaped=None, tc_reshaped=None, num_chans=62, tf=128, tc=40, use_coarse_time="B"):
    """
    Invert the chop_and_reshape_signals operation.
    use_coarse_time must match what was used there.

    Test it out with this example:
        tf = 16
        tc = 10
        num_chans = 21
        #
        mc = torch.zeros(num_chans,tf*tc)   # Labeled Channels
        mt = torch.zeros(num_chans,tf*tc)   # Labeled time_pts
        cp = torch.zeros(num_chans,3)       # Labeled Channel {x,y,z}-positions
        #
        for i in range(num_chans):
            cp[i,0] = i + 0.0       # label for x
            cp[i,1] = i + 0.1       # label for y
            cp[i,2] = i + 0.2       # label for z
            for j in range(tf*tc):
                mc[i,j] = i
                mt[i,j] = j
        #
        nc, cpr, cpdr, cir, tcr, sql = chop_and_reshape_signals(eeg_signal=mc, chan_pos=cp, chan_pos_discrete=cp, tf=tf, use_coarse_time="B"|"A"|"C")
        nt, cpr, cpdr, cir, tcr, sql = chop_and_reshape_signals(eeg_signal=mt, chan_pos=cp, chan_pos_discrete=cp, tf=tf, use_coarse_time="B"|"A"|"C")

        # inspect nc, nt, cpr, cpdr, cir, tcr, sql

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -     

        oc, cpu, cpdu, ciu, tcu = invert_reshape_signals(sig_reshaped=nc, pos_reshaped=cpr, pos_discrete_reshaped=cpdr, id_reshaped=cir, tc_reshaped=tcr, num_chans=num_chans, tf=tf, use_coarse_time="B"|"A"|"C")
        ot, cpu, cpdu, ciu, tcu = invert_reshape_signals(sig_reshaped=nt, pos_reshaped=cpr, pos_discrete_reshaped=cpdr, id_reshaped=cir, tc_reshaped=tcr, num_chans=num_chans, tf=tf, use_coarse_time="B"|"A"|"C")  

        # 1. Assert that the unwrapping and reshaping of signal worked correctly: inspect oc & ot (should match mc & mt)
        assert (otB==mt).all().item()
        assert (ocB==mc).all().item()
        # 2. Assert that the unwrapping and reshaping of channel positions worked correctly: shape = [num_chans, tc, 3]
        mod_in_pos_unwrapt = cpu
        chan_pos = mod_in_pos_unwrapt.reshape(-1,tc,3)
        for k in range(num_chans):
            tc0 = chan_pos[k,0,:]
            for j in range(1, tc):
                assert (tc0 == chan_pos[k,j,:]).all().item(), f"chan_pos unwrapping not right for sample {k}, time {j}."
        # 3. Assert that the unwrapping and reshaping for channel id worked correctly: shape = [num_chans, tc]
        chan_id_unwrapt = ciu
        for k in range(num_chans):
            assert (chan_id_unwrapt[k]==k).all().item(), f"chan_id unwrapping {k} not right."
        # 4. Assert that the unwrapping and reshaping for coarse_time worked correctly: shape = [num_chan, tc]
        tc_unwrapt = tcu
        if tc_unwrapt is not None:
            tc0 = tc_unwrapt[0]
            for j in range(1, num_chans):
                assert (tc0 == tc_unwrapt[j]).all().item(), f"coarse time unwrapping {j} not right."

    """

    # tc = sig_reshaped.shape[0]//num_chans
    num_tpts = tc*tf

    if use_coarse_time=="A":
        # Keep same coarse-time values together in reshaping.
        sig_unwrapt = sig_reshaped.reshape(tc, num_chans, tf).transpose(0,1).reshape(num_chans,num_tpts) if sig_reshaped is not None else None
        pos_unwrapt = pos_reshaped.reshape(tc, num_chans, 3).transpose(0,1).reshape(num_chans,3*tc) if pos_reshaped is not None else None
        pos_discrete_unwrapt = pos_discrete_reshaped.reshape(tc, num_chans, 3).transpose(0,1).reshape(num_chans,3*tc) if pos_discrete_reshaped is not None else None
        id_unwrapt = id_reshaped.reshape(tc, num_chans).T if id_reshaped is not None else None
        tc_unwrapt = tc_reshaped.reshape(tc, num_chans).T if tc_reshaped is not None else None 

    elif use_coarse_time=="B":
        # Keep same channels together in reshaping
        sig_unwrapt = sig_reshaped.reshape(tc, num_chans, tf).reshape(num_chans,num_tpts) if sig_reshaped is not None else None
        pos_unwrapt = pos_reshaped.reshape(tc, num_chans, 3).reshape(num_chans,3*tc) if pos_reshaped is not None else None
        pos_discrete_unwrapt = pos_discrete_reshaped.reshape(tc, num_chans, 3).reshape(num_chans,3*tc) if pos_discrete_reshaped is not None else None
        id_unwrapt = id_reshaped.reshape(num_chans, tc) if id_reshaped is not None else None
        tc_unwrapt = tc_reshaped.reshape(num_chans, tc) if tc_reshaped is not None else None 

    elif use_coarse_time=="C":
        # Just use first tf timepoints of each channel's eeg signal.
        sig_unwrapt = sig_reshaped 
        pos_unwrapt = pos_reshaped 
        pos_discrete_unwrapt = pos_discrete_reshaped 
        id_unwrapt = id_reshaped 
        tc_unwrapt = tc_reshaped 

    elif use_coarse_time=="D":
        # Single channel for tc=10
        num_chans=1
        sig_unwrapt = sig_reshaped.reshape(tc, num_chans, tf).reshape(num_chans,num_tpts) if sig_reshaped is not None else None
        pos_unwrapt = pos_reshaped.reshape(tc, num_chans, 3).reshape(num_chans,3*tc) if pos_reshaped is not None else None
        pos_discrete_unwrapt = pos_discrete_reshaped.reshape(tc, num_chans, 3).reshape(num_chans,3*tc) if pos_discrete_reshaped is not None else None
        id_unwrapt = id_reshaped.reshape(num_chans, tc) if id_reshaped is not None else None
        tc_unwrapt = tc_reshaped.reshape(num_chans, tc) if tc_reshaped is not None else None 

    else:
        print(f"Not Implemented Error: {use_coarse_time=} and it needs to be A, B, C or D.")
        die


    return sig_unwrapt, pos_unwrapt, pos_discrete_unwrapt, id_unwrapt, tc_unwrapt   



@dataclass
class BCIDatasetArgs:
    use_b2: bool = False # If true, use Backblaze B2 for dataset loading, otherwise use local filesystem.
    data_dir: str = "/data/groups/bci/datasets/v7_train/"
    export_dir: str = "" # Where to save output .pt files after inference.
    glob_filter: List[str] = field(default_factory=lambda: ["**/*.pt"]) # default is to use all .pt files in all subdirectories.
    chan_num_filter: Union[int, None] = None # None or integer number of channels we want in each sample
    sample_rate: int = 256 # 512 # Passing in from config now.
    seq_len: int = 1280 # 2560 # Passing in from config now.
    num_fine_time_pts: int = 128
    use_coarse_time: str = "B" # How to chop signals in to coarse-time, fine-time & channels using chop_and_reshape_signals or chop_signals_only
    cat_chan_xyz_and_eeg: bool = False #True - havent used in a while. Default to False
    dont_noise_chan_xyz: bool = False # If true, do not add noise to channel {x,y,z}-position in EEGProcessor.process (use in tandem with NoPE)
    randomly_permute_sequence: bool = False

    data_norm: float = 1.0 # The norm to divide the data by, to normalize it to [-1,1] range.
    data_clip: float = 1.0 # Clip data to this value after normalization.

    sample_duration_seconds: float = 5.0

    min_sample_duration_seconds: float = 0.25 # seconds
    max_sample_duration_seconds: float = 30.0 # seconds

    num_batches: Union[int, None] = None

    # CLODE fixed-eval harness (see eval_harness_clode.md)
    fixed_eval: bool = False                       # replay a frozen, sharded pool instead of streaming random draws
    eval_noise_seed: int = 0                       # base seed; per-sample seed = eval_noise_seed + global_pool_idx
    fixed_eval_cache_dir: Union[str, None] = None  # override where frozen_eval_*.pt is stored (default: sibling of data_dir)
    plot_num_batches: int = 5                      # eeg_eval.py: how many frozen samples to plot + score (subset of num_batches)

    crop_size: Union[int, None] = None

    encoder_input_channels: int = 64 # NOT USING ANYLONGER. GET RID OF.
    decoder_input_channels: int = 64 # NOT USING ANYLONGER. GET RID OF.
    token_dropout_prob: int | float = -1.0 # Probability of applying channel dropout (negative to turn off)
    dropout_scheme: str = "train-2" # {"train-1", "train-2", "eval-1"}

    batch_size: int = 32
    target_packed_seqlen: int =  16384
    do_N_epochs: Union[int, None] = None
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: Union[int, None] = 2
    shuffle: bool = True
    seed: Union[int, None] = None

    diffusion_noise_schedule: str = "linear"   # {"linear","beta","logit"}
    logit_normal_mean: float = 0.0   # if diffusion_noise_schedule==logit, centre of hump = sigmoid(mean); 0 -> t=0.5
    logit_normal_std:  float = 1.0   # if diffusion_noise_schedule==logit, width; ~1 unimodal hump (SD3), >=2 -> U-shaped

    pad_packed_seqlen: bool = False  # CLODE: if True, pad each packed seq with one all-zero document up to EXACTLY
                                     # target_packed_seqlen (fixed shapes -> no torch.compile recompiles / frag).
                                     # Requires target_packed_seqlen % encoder_latent_downsample_factor == 0.

    diffusion_forcing: bool = False
    diffusion_forcing_num_frames: int = 1

    patching_type: str = "frames"
    stft_global_sigma: Union[str, float] = 1.0
    masked_in_decoder: bool = True # If true, mask out channels in decoder input when channel is dropped. (true works, false does not)

    num_bins_discretize_xyz_chan_pos: int = 100 # Number of bins to discretize channel positions to use in 4d-RoPE. # 40 with "old" xyz_extremes, 100 with "thirteens" xyz_extremes
    chan_pos_xyz_extremes_type: str = "thirteens" # "old" for v4 dataset or "thirteens" for v5 dataset

    # v3 mmap fields — ignored by EEGDataset_v2/b2, used only when use_v3=True  #jm
    use_v3: bool = False                                  #jm
    filter_version: List[str] = field(default_factory=lambda: ["v3_bandpass"]) # WAS str = "v3_bandpass"                   #jm  (use_v3: reads mmap from data_dir)
    min_quality_any: float = 0.1                          #jm
    min_quality_mean: float = 0.3                         #jm
    dataset_id: int = 7                                   #jm    
    sample_duration_str: str = "5_seconds" # {"5_seconds", "10_seconds", "30_seconds"}
    do_avg_ref: bool = True # If true, do average reference before data normalization.
    z_score_type: str = "across_sample" # {"across_channel", "across_sample", "none"}
    mmap_sample_start: None|int = None # If not None, only sample from between this and stop in the mmap.
    mmap_sample_stop: None|int = None # If not None, only sample up to this and start in the mmap.
    skip_preepoched_data: bool = False # If true, skip pre-epoched data.
    
    # Backblaze B2 specific fields (for EEGDataset_b2)
    load_dotenv()
    b2_bucket_name: Optional[str] = "zyphra-bci" #None # e.g., "zyphra-bci"
    b2_endpoint_url: Optional[str] = "https://s3.us-west-004.backblazeb2.com" #None  # e.g., "https://s3.us-west-000.backblazeb2.com"
    b2_access_key_id: Optional[str] = os.getenv("B2_ACCESS_KEY_ID") #None
    b2_secret_access_key: Optional[str] = os.getenv("B2_SECRET_ACCESS_KEY") #None
    b2_local_cache_dir: Optional[str] = "/mnt/shared/datasets/bci/b2_cache"  # Local directory to cache downloaded files
    b2_cache_files: bool = False  # Whether to cache files locally or download on-demand



def discretize_chan_pos(chan_pos, xyz_extremes, num_bins):
    """
    Discretize continuous channel positions into integer bins.

    Args:
        chan_pos: Tensor of shape [num_channels, 3] with continuous (x, y, z) positions
        xyz_extremes: Tensor of shape [2, 3] where xyz_extremes[0] is min values
                      and xyz_extremes[1] is max values for each dimension
        num_bins: Integer number of bins to use for discretization

    Returns:
        chan_pos_discrete: Tensor of shape [num_channels, 3] with integer bin indices
    """


    # Extract min and max values for each dimension
    xyz_min = xyz_extremes[0]  # shape: [3]
    xyz_max = xyz_extremes[1]  # shape: [3]

    # Check if all positions are within the specified min/max bounds
    within_min = (chan_pos >= xyz_min).all()
    within_max = (chan_pos <= xyz_max).all()

    if not (within_min and within_max):
        import warnings
        out_of_bounds_min = chan_pos < xyz_min
        out_of_bounds_max = chan_pos > xyz_max
        warnings.warn(
            f"Channel positions out of bounds detected!\n"
            f"  Positions below min: {out_of_bounds_min.sum().item()} elements\n"
            f"  Positions above max: {out_of_bounds_max.sum().item()} elements\n"
            f"  xyz_min: {xyz_min.tolist()}\n"
            f"  xyz_max: {xyz_max.tolist()}\n"
            f"  chan_pos range: [{chan_pos.min(dim=0).values.tolist()}, {chan_pos.max(dim=0).values.tolist()}]"
        )

    # Normalize channel positions to [0, 1] range
    chan_pos_normalized = (chan_pos - xyz_min) / (xyz_max - xyz_min)

    # Scale to [0, num_bins) and convert to integer bin indices
    chan_pos_discrete = (chan_pos_normalized * num_bins).long()

    # Clamp values to ensure they're within valid range [0, num_bins-1]
    chan_pos_discrete = torch.clamp(chan_pos_discrete, 0, num_bins - 1)

    return chan_pos_discrete





def perform_token_dropout(dropout_scheme, token_dropout_prob, num_fine_time_pts, mmap, channel_names=None, chan_pos=None):
    """
    Perform token dropout on a mmap.
    Options for dropout_scheme:
        - "train-1": channel dropout
        - "train-2": full-channel-random-dropout-train
        - "random-uniform-dropout": random-uniform-dropout
        - "full-time-pt-random-dropout": full-time-pt-random-dropout
        - "correlated-channel-time-dropout": correlated-channel-time-dropout
        - "mix-4-dropouts-train": mix-4-dropouts-train
        - "mix-7-dropouts-train": mix-7-dropouts-train
        - "spatially-selective-dropout": spatially-selective-dropout
        - "consumer-eeg-channel-dropout": consumer-eeg-channel-dropout
        - "standard-montage-channel-dropout": standard-montage-channel-dropout
        - "brain-region-channel-dropout": brain-region-channel-dropout
        - "eval-1": eval-1
        - "full-channel-random-dropout-eval": full-channel-random-dropout-eval
    """

    # Sample which dropout scheme to use with 1/N probability
    if dropout_scheme == "mix-4-dropouts-train":
        dropout_scheme = random.choices([
            "random-uniform-dropout", 
            "full-channel-random-dropout-train", 
            "full-time-pt-random-dropout", 
            "correlated-channel-time-dropout"], 
            weights=[0.25, 0.25, 0.25, 0.25])[0]

    elif dropout_scheme == "mix-3-dropouts-train":
        dropout_scheme = random.choices([
            "random-uniform-dropout", 
            "full-channel-random-dropout-train", 
            "correlated-channel-time-dropout"], 
            weights=[0.33, 0.33, 0.33])[0]

    elif dropout_scheme == "mix-3-position-dropouts-train": # temporary dropout scheme for position-based dropouts.
        dropout_scheme = random.choices([
            "consumer-eeg-channel-dropout", 
            "standard-montage-channel-dropout", 
            "brain-region-channel-dropout"], 
            weights=[0.33, 0.33, 0.33])[0]

    elif dropout_scheme == "mix-8-dropouts-train":
        dropout_scheme = random.choices([
            "standard-montage-channel-dropout",
            "random-uniform-dropout", # too easy. downweight.
            "full-channel-random-dropout-train", 
            "correlated-channel-time-dropout",
            "full-time-pt-random-dropout", 
            "random-montage-channel-dropout", 
            "brain-region-channel-dropout",
            "consumer-eeg-channel-dropout"], # too hard. downweight.
            weights=[0.125, 0.075, 0.275, 0.125, 0.125, 0.125, 0.125, 0.025])[0]

    else:
        dropout_scheme = dropout_scheme


    if dropout_scheme == "train-1":
        ## NOTE: THIS WAS OUR FIRST DROPOUT SCHEME USED FOR TRAINING - FOR TEST69 TO TEST83
        # Apply channel dropout right here to get list of channels to drop
        token_dropout = []
        for mm in mmap:
            if random.random() < token_dropout_prob:
                N = mm.shape[0]
                if N<=1: # if there is only 1 channel, cannot dropout any.
                    token_dropout.append([]) # No dropout for this sample.
                    continue
                M = random.randint(1, N-1)
                random_integers = sorted(random.sample(range(1, N), M))
                token_dropout.append(random_integers)
            else:
                token_dropout.append([]) # No dropout for this sample.

    elif dropout_scheme == "full-channel-random-dropout-train" or dropout_scheme == "train-2":
        ## NOTE: USING THIS IMPROVED DROPOUT SCHEME USED FOR TRAINING - STARTING WITH TEST84 - TRYING OUT THERE.
        # Apply NEW channel dropout right here to get list of all tokens (ch,tc) to drop
        #   a. self.token_dropout_prob determines whether we do channel dropout for this sample.
        #   If we do channel dropout, 
        #       b. with p=0.8, we drop between 1 and N/2 chans with uniform probability.
        #       c. with p=0.2, we drop between N/2 and N-1 chans with uniform probability.
        token_dropout = []
        for mm in mmap:
            if random.random() < token_dropout_prob:
                N,T = mm.shape
                tc = T/num_fine_time_pts
                if tc%1 == 0:
                    tc_list = list(range(int(tc))) # list of coarse-time indices
                else:
                    print(f"Inside perform_token_dropout, Dropout scheme: {dropout_scheme}, Warning: {tc=} is not an integer!")

                if N<=1: # if there is only 1 channel, cannot dropout any.
                    token_dropout.append([]) # No dropout for this sample.
                    continue
                rand_num = random.random()
                if rand_num < 0.6 and N//4 > 1: # 60% of the time, drop between 1 and N/4 channels (if N//4 > 1)
                    M = random.randint(1, N//4)
                elif rand_num < 0.9: # 30% of the time, drop between N/4 and N/2 channels
                    M = random.randint(N//4, N//2)
                else: # 10% of the time, drop between N/2 and N-1 channels
                    M = random.randint(N//2, N-1)
                random_integers = sorted(random.sample(range(1, N), M)) # channels to drop
                combined_coords = [(r, t) for r in random_integers for t in tc_list] # coords (chan, coarse-time) to drop
                token_dropout.append(combined_coords)
            else:
                token_dropout.append([]) # No dropout for this sample.

    elif dropout_scheme == "random-uniform-dropout":
        # Randomly and independently drop out (prob*chans*T) spots in the data matrix in each sample.
        #
        token_dropout = []
        for mm in mmap:
            if random.random() < token_dropout_prob:
                if random.random() < 0.2:
                    M = random.uniform(0.1, 0.5)
                else:
                    M = random.uniform(0.5, 0.9)
                ch, T = mm.shape
                num_to_drop = int(M * ch * T)
                flat = random.sample(range(ch * T), num_to_drop)
                coords = [(i % ch, i // ch) for i in flat]
                token_dropout.append(coords)
            else:
                token_dropout.append([]) # No dropout for this sample.
        
    elif dropout_scheme == "full-time-pt-random-dropout":
        # Apply time-point dropout right here to get list of all tokens (ch,tc) to drop
        #   a. self.token_dropout_prob determines whether we do time-point dropout for this sample.
        #   If we do time-point dropout, 
        #       b. Draw tc_width from a triangle distribution defined by low, mode, high.  High is constrained to be no more than 80% of sample
        #       c. Draw tc_begin randomly between 0 and out a section of tc_width width centered at a random tc index.

        token_dropout = []
        for mm in mmap:
            if random.random() < token_dropout_prob:
                N,T = mm.shape
                ch_list = list(range(N)) # list of channels in sample
                tc_max = T/num_fine_time_pts # number of coarse-time points in sample
                if tc_max%1 == 0:
                    tc_max = int(tc_max)
                else:
                    print(f"Inside perform_token_dropout, Dropout scheme: {dropout_scheme}, Warning: {tc_max=} is not an integer!")
                
                # Sample amount of time points to drop .
                if random.random() < 1.1: #0.8: # 100% of the time !!
                    tc_stop_thresh = random.randint(int(0.1*tc_max), int(0.2*tc_max))
                else:
                    tc_stop_thresh = random.randint(int(0.25*tc_max), int(0.5*tc_max))

                tc_list = set() # list of lists of tc indices to drop
                cnt = 0
                tc_buffer = 1 # make sure dropped tokens arent at exact beginning or end of sample.
                while len(tc_list) < tc_stop_thresh:
                    # Expand the list of tc indices to drop by 1 time point on each side (so we dont't long contiguous time points).
                    tc_plus = {x + 1 for x in tc_list}
                    tc_minus = {x - 1 for x in tc_list}
                    tc_expand = tc_list.union(tc_plus).union(tc_minus)
                    #
                    # Distribution of tc width of section to dropout: low, high, mode (the peak)
                    low, mode, high = 2, 4, min(16, int(0.2*tc_max)) # in units of tc (num_fine_time_pts/sample_rate) - 0.125s
                    
                    tc_width = int(np.round(random.triangular(low, high, mode)))
                    tc_begin = random.randint(tc_buffer, tc_max - tc_width - tc_buffer)
                    tc_to_add = list(range(tc_begin, tc_begin + tc_width))
                    if set(tc_to_add).isdisjoint(tc_expand):
                        tc_list.update(tc_to_add)
                    cnt+=1
                    if cnt > 3: #5: # 30:
                        break


                combined_coords = [(c, t) for c in ch_list for t in tc_list] # coords (chan, coarse-time) to drop
                token_dropout.append(combined_coords) 
            else:
                token_dropout.append([]) # No dropout for this sample.
                                

    elif dropout_scheme == "correlated-channel-time-dropout":
        # Apply correlated channel + time-point dropout right here to get list of all tokens (ch,tc) to drop
        # THIS BASICALLY COMBINES THE FULL-TIME-PT-RANDOM-DROPOUT SCHEME WITH THE FULL-CHANNEL-RANDOM-DROPOUT SCHEME.
        #   a. self.token_dropout_prob determines whether we do time-point dropout for this sample.
        #   If we do correlated channel + time-point dropout, 
        #       b. Draw tc_width from a triangle distribution defined by low, mode, high.  High is constrained to be no more than 80% of sample
        #       c. Draw tc_begin randomly between 0 and out a section of tc_width width centered at a random tc index.
        #       d. with p=0.8, we drop between 1 and N/2 chans with uniform probability.
        #       e. with p=0.2, we drop between N/2 and N-1 chans with uniform probability.

        token_dropout = []
        for mm in mmap:
            if random.random() < token_dropout_prob:
                N,T = mm.shape
                tc_max = T/num_fine_time_pts # number of coarse-time points in sample
                if tc_max%1 == 0:
                    tc_max = int(tc_max)
                else:
                    print(f"Inside perform_token_dropout, Dropout scheme: {dropout_scheme}, Warning: {tc_max=} is not an integer!")

                # Sample amount of time points to drop .
                if random.random() < 0.5:
                    tc_stop_thresh = random.randint(int(0.1*tc_max), int(0.5*tc_max))
                else:
                    tc_stop_thresh = random.randint(int(0.5*tc_max), int(0.9*tc_max))

                tc_list = set() # list of lists of tc indices to drop
                cnt = 0
                tdo_inner = []
                while len(tc_list) < tc_stop_thresh:
                    # Expand the list of tc indices to drop by 1 time point on each side (so we dont't long contiguous time points).
                    tc_plus = {x + 1 for x in tc_list}
                    tc_minus = {x - 1 for x in tc_list}
                    tc_expand = tc_list.union(tc_plus).union(tc_minus)
                    #
                    # Distribution of tc width of section to dropout: low, high, mode (the peak)
                    low, mode, high = 2, 4, min(16, int(0.8*tc_max)) # in units of tc (num_fine_time_pts/sample_rate) - 0.125s
                    tc_width = int(np.round(random.triangular(low, high, mode)))
                    tc_begin = random.randint(0, tc_max - tc_width)
                    tc_to_add = list(range(tc_begin, tc_begin + tc_width))
                    if set(tc_to_add).isdisjoint(tc_expand):
                        tc_list.update(tc_to_add)

                        if N<=1: # if there is only 1 channel, cannot dropout any.
                            tdo_inner.extend([]) # No dropout for this sample.
                            continue

                        if random.random() < 0.5:
                            M = random.randint(1, N//2)
                        else:
                            M = random.randint(N//2, N-1)
                        ch_list = sorted(random.sample(range(1, N), M)) # channels to drop
                        combined_coords = [(c, t) for c in ch_list for t in tc_to_add] # coords (chan, coarse-time) to drop
                        tdo_inner.extend(combined_coords) 

                    cnt+=1
                    if cnt > 50:
                        break

                token_dropout.append(tdo_inner)
            else:
                token_dropout.append([]) # No dropout for this sample.

    elif dropout_scheme == "brain-region-channel-dropout":
        # Assign a brain region to each channel from xyz coordinates (metres).
        # x=right(+), y=front(+), z=up; values expected in metres (~±0.09 m).
        # Tuned for balance (~3% std across regions on TUH/ONE/CW v7 sample) —
        # mirrors threshold_rejection_analysis.py:_xyz_to_region.
        def _xyz_to_region(xyz_m):
            x, y, z = float(xyz_m[0]) * 1000, float(xyz_m[1]) * 1000, float(xyz_m[2]) * 1000
            hemi = "left" if x <= 0 else "right"
            if y > 35:
                return f"frontal_{hemi}"
            if y < -55:
                return f"occipital_{hemi}"
            if abs(x) > 60 and -55 <= y <= 35:
                return f"temporal_{hemi}"
            if -55 <= y < -15:
                return "parietal"
            return "central"

        token_dropout = []
        for mm in mmap:
            if random.random() < token_dropout_prob:
                n_ch = mm.shape[0]

                if chan_pos is None:
                    token_dropout.append([])
                    continue

                xyz_np = np.array(chan_pos, dtype=float)
                if np.abs(xyz_np).max() > 1.0:
                    xyz_np = xyz_np / 1000.0
                channel_regions = [_xyz_to_region(xyz_np[i]) for i in range(n_ch)]

                present_regions = list({r for r in channel_regions if r is not None})

                #
                if len(present_regions) < 2:
                    token_dropout.append([])
                    continue

                iter_count = 0
                while True:
                    k = random.randint(len(present_regions)//2, len(present_regions) - 1) # bias towards keeping more regions.
                    chosen_regions = set(random.sample(present_regions, k))
                    channels_to_drop = sorted([
                        i for i, r in enumerate(channel_regions)
                        if r not in chosen_regions
                    ])
                    iter_count += 1
                    if iter_count > 30:
                        channels_to_drop = []
                        break
                    if 3 < len(channels_to_drop) < n_ch - 3:
                        break

                N,T = mm.shape
                tc = T/num_fine_time_pts
                if tc%1 == 0:
                    tc_list = list(range(int(tc))) # list of coarse-time indices
                else:
                    print(f"Inside perform_token_dropout, Dropout scheme: {dropout_scheme}, Warning: {tc=} is not an integer!")

                combined_coords = [(r, t) for r in channels_to_drop for t in tc_list] # coords (chan, coarse-time) to drop
                token_dropout.append(combined_coords)

            else:
                token_dropout.append([]) # No dropout for this sample.





    elif dropout_scheme == "brain-region-channel-dropout-old":
        # Assign a brain region to each channel from xyz coordinates (metres).
        # x=right(+), y=front(+), z=up; values expected in metres (~±0.09 m).
        def _xyz_to_region(xyz_m):
            x, y, z = float(xyz_m[0]) * 1000, float(xyz_m[1]) * 1000, float(xyz_m[2]) * 1000
            hemi = "left" if x < -20 else ("right" if x > 20 else "mid")
            if y > 45 or (z < 10 and y > 15):
                base = "frontal"
            elif y < -60 or (z < 10 and y < -35):
                base = "occipital"
            elif abs(x) > 60 and abs(y) < 40:
                base = "temporal"
            elif y < -25 and z > 20:
                base = "parietal"
            elif abs(x) < 55 and abs(y) < 32 and z > 35:
                base = "central"
            else:
                return None
            if base == "frontal":
                return "frontal_right" if hemi == "right" else "frontal_left"
            if base == "temporal":
                return f"temporal_{'left' if hemi == 'left' else 'right'}" if hemi != "mid" else None
            return base

        token_dropout = []
        for mm in mmap:
            if random.random() < token_dropout_prob:
                n_ch = mm.shape[0]

                if chan_pos is None:
                    token_dropout.append([])
                    continue

                xyz_np = np.array(chan_pos, dtype=float)
                if np.abs(xyz_np).max() > 1.0:
                    xyz_np = xyz_np / 1000.0
                channel_regions = [_xyz_to_region(xyz_np[i]) for i in range(n_ch)]

                present_regions = list({r for r in channel_regions if r is not None})
                if len(present_regions) < 2:
                    token_dropout.append([])
                    continue

                iter_count = 0
                while True:
                    k = random.randint(1, len(present_regions) - 1)
                    chosen_regions = set(random.sample(present_regions, k))
                    channels_to_drop = sorted([
                        i for i, r in enumerate(channel_regions)
                        if r not in chosen_regions
                    ])
                    iter_count += 1
                    if iter_count > 30:
                        channels_to_drop = []
                        break
                    if 3 < len(channels_to_drop) < n_ch - 3:
                        break

                N,T = mm.shape
                tc = T/num_fine_time_pts
                if tc%1 == 0:
                    tc_list = list(range(int(tc))) # list of coarse-time indices
                else:
                    print(f"Inside perform_token_dropout, Dropout scheme: {dropout_scheme}, Warning: {tc=} is not an integer!")

                combined_coords = [(r, t) for r in channels_to_drop for t in tc_list] # coords (chan, coarse-time) to drop
                token_dropout.append(combined_coords)

            else:
                token_dropout.append([]) # No dropout for this sample.



    elif dropout_scheme == "brain-region-channel-dropout-by-name":
        # Dropout channels by brain region name.
        #
        REGION_CHANNELS = {
            "frontal_left":   {"fp1", "fpz", "af3", "af7", "afz","f1", "f3", "f5",
                               "f7", "fz","fc1", "fc3", "fc5", "fcz"},
            "frontal_right":  {"fp2", "af4", "af8", "f2", "f4", "f6", "f8",
                               "fc2", "fc4", "fc6"},
            "temporal_left":  {"t3", "t5", "t7", "ft7", "tp7", "tp9"},
            "temporal_right": {"t4", "t6", "t8", "ft8", "tp8", "tp10"},
            "central":        {"c1", "c2", "c3", "c4", "c5", "c6", "cz"},
            "parietal":       {"p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "pz",
                               "cp1", "cp2", "cp3", "cp4", "cp5", "cp6", "cpz"},
            "occipital":      {"o1", "o2", "oz", "po3", "po4", "po7", "po8", "poz",
                               "i1", "i2", "iz"},
        }

        token_dropout = []
        for mm in mmap:
            if random.random() < token_dropout_prob:
                iter_count = 0
                while True:
                    # Choose between 1 and n_keys - 1 regions to keep.
                    n_keys = len(REGION_CHANNELS)
                    k = random.randint(1, n_keys - 1)
                    chosen_regions = random.sample(list(REGION_CHANNELS.keys()), k)
                    channels_to_keep = set().union(*(REGION_CHANNELS[rk] for rk in chosen_regions))
                    #
                    channels_to_drop = sorted([
                        i for i, name in enumerate(channel_names)
                        if name not in channels_to_keep
                    ])
                    iter_count += 1
                    if iter_count > 30:
                        channels_to_drop = []
                        break

                    # sample headset again if we don't have any channels to drop or we drop all channels
                    if 3 < len(channels_to_drop) < len(channel_names)-3:
                        break

                N,T = mm.shape
                tc = T/num_fine_time_pts
                if tc%1 == 0:
                    tc_list = list(range(int(tc))) # list of coarse-time indices
                else:
                    print(f"Inside perform_token_dropout, Dropout scheme: {dropout_scheme}, Warning: {tc=} is not an integer!")

                combined_coords = [(r, t) for r in channels_to_drop for t in tc_list] # coords (chan, coarse-time) to drop
                token_dropout.append(combined_coords)

            else:
                token_dropout.append([]) # No dropout for this sample.


    elif dropout_scheme == "random-montage-channel-dropout":
        # Greedily prune nearest-neighbour pairs until a target count
        # (8, 16, 32, or 64) is reached, giving sparse but global coverage.
        _TARGET_COUNTS = [8, 16, 32, 64]

        token_dropout = []
        for mm in mmap:
            if random.random() < token_dropout_prob:
                n_ch = mm.shape[0]

                if chan_pos is None:
                    token_dropout.append([])
                    continue

                xyz_np = np.array(chan_pos, dtype=float)
                valid_targets = [t for t in _TARGET_COUNTS if t < n_ch - 3]
                if not valid_targets:
                    token_dropout.append([])
                    continue

                weights = [t*t for t in valid_targets] # weight larger targets more
                target = random.choices(valid_targets, weights=weights, k=1)[0]

                # Greedily drop the channel that is closest to any other channel
                kept = list(range(n_ch))
                while len(kept) > target:
                    pos = xyz_np[kept]
                    dists = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=-1))
                    np.fill_diagonal(dists, np.inf)
                    i, j = divmod(int(np.argmin(dists)), len(kept))
                    kept.pop(random.choice([i, j]))
                channels_to_drop = sorted(set(range(n_ch)) - set(kept))

                if not (3 < len(channels_to_drop) < n_ch - 3):
                    token_dropout.append([])
                    continue

                N,T = mm.shape
                tc = T/num_fine_time_pts
                if tc%1 == 0:
                    tc_list = list(range(int(tc))) # list of coarse-time indices
                else:
                    print(f"Inside perform_token_dropout, Dropout scheme: {dropout_scheme}, Warning: {tc=} is not an integer!")

                combined_coords = [(r, t) for r in channels_to_drop for t in tc_list] # coords (chan, coarse-time) to drop
                token_dropout.append(combined_coords)

            else:
                token_dropout.append([]) # No dropout for this sample.



    elif dropout_scheme == "standard-montage-channel-dropout":
        # Standard 10-20/10-10 xyz positions (metres) used as target locations for each standard montage.
        # For each target position, the nearest actual channel is kept; all others are dropped.
        _STD_XYZ = {
            "fp1":  (-0.026,  0.083,  0.020), "fp2":  ( 0.026,  0.083,  0.020),
            "fpz":  ( 0.000,  0.087,  0.020),
            "af7":  (-0.068,  0.065,  0.015), "af8":  ( 0.068,  0.065,  0.015),
            "af3":  (-0.040,  0.071,  0.048), "af4":  ( 0.040,  0.071,  0.048),
            "afz":  ( 0.000,  0.073,  0.060),
            "f7":   (-0.083,  0.048,  0.012), "f8":   ( 0.083,  0.048,  0.012),
            "f5":   (-0.067,  0.050,  0.046), "f6":   ( 0.067,  0.050,  0.046),
            "f3":   (-0.047,  0.052,  0.063), "f4":   ( 0.047,  0.052,  0.063),
            "f1":   (-0.024,  0.054,  0.072), "f2":   ( 0.024,  0.054,  0.072),
            "fz":   ( 0.000,  0.054,  0.074),
            "ft7":  (-0.087,  0.025,  0.012), "ft8":  ( 0.087,  0.025,  0.012),
            "fc5":  (-0.073,  0.026,  0.052), "fc6":  ( 0.073,  0.026,  0.052),
            "fc3":  (-0.052,  0.026,  0.073), "fc4":  ( 0.052,  0.026,  0.073),
            "fc1":  (-0.026,  0.026,  0.085), "fc2":  ( 0.026,  0.026,  0.085),
            "fcz":  ( 0.000,  0.026,  0.087),
            "t7":   (-0.090,  0.000,  0.010), "t8":   ( 0.090,  0.000,  0.010),
            "c5":   (-0.078,  0.000,  0.046), "c6":   ( 0.078,  0.000,  0.046),
            "c3":   (-0.054,  0.000,  0.073), "c4":   ( 0.054,  0.000,  0.073),
            "c1":   (-0.027,  0.000,  0.087), "c2":   ( 0.027,  0.000,  0.087),
            "cz":   ( 0.000,  0.000,  0.090),
            "tp7":  (-0.087, -0.025,  0.012), "tp8":  ( 0.087, -0.025,  0.012),
            "cp5":  (-0.073, -0.026,  0.052), "cp6":  ( 0.073, -0.026,  0.052),
            "cp3":  (-0.052, -0.026,  0.073), "cp4":  ( 0.052, -0.026,  0.073),
            "cp1":  (-0.026, -0.026,  0.085), "cp2":  ( 0.026, -0.026,  0.085),
            "cpz":  ( 0.000, -0.026,  0.087),
            "p7":   (-0.083, -0.048,  0.012), "p8":   ( 0.083, -0.048,  0.012),
            "p5":   (-0.067, -0.050,  0.046), "p6":   ( 0.067, -0.050,  0.046),
            "p3":   (-0.047, -0.052,  0.063), "p4":   ( 0.047, -0.052,  0.063),
            "p1":   (-0.024, -0.054,  0.072), "p2":   ( 0.024, -0.054,  0.072),
            "pz":   ( 0.000, -0.054,  0.074),
            "po7":  (-0.068, -0.065,  0.015), "po8":  ( 0.068, -0.065,  0.015),
            "po3":  (-0.040, -0.071,  0.048), "po4":  ( 0.040, -0.071,  0.048),
            "poz":  ( 0.000, -0.073,  0.060),
            "o1":   (-0.026, -0.083,  0.020), "o2":   ( 0.026, -0.083,  0.020),
            "oz":   ( 0.000, -0.087,  0.020),
        }

        _STANDARD_MONTAGES_XYZ = {
            "standard_8":  ["fp1", "fp2", "c3", "cz", "c4", "o1", "o2", "pz"],
            "standard_16": ["fp1", "fp2",
                            "f3", "fz", "f4",
                            "c3", "cz", "c4",
                            "t7", "t8",
                            "p3", "pz", "p4",
                            "o1", "oz", "o2"],
            "standard_32": ["fp1", "fp2", "fpz",
                            "af3", "af4",
                            "f7", "f3", "fz", "f4", "f8",
                            "fc5", "fc1", "fcz", "fc2", "fc6",
                            "t7", "c3", "cz", "c4", "t8",
                            "cp5", "cp1", "cpz", "cp2", "cp6",
                            "p7", "p3", "pz", "p4", "p8",
                            "o1", "oz", "o2"],
            "standard_64": ["fp1", "fp2", "fpz",
                            "af7", "af3", "afz", "af4", "af8",
                            "f7", "f5", "f3", "f1", "fz", "f2", "f4", "f6", "f8",
                            "ft7", "fc5", "fc3", "fc1", "fcz", "fc2", "fc4", "fc6", "ft8",
                            "t7", "c5", "c3", "c1", "cz", "c2", "c4", "c6", "t8",
                            "tp7", "cp5", "cp3", "cp1", "cpz", "cp2", "cp4", "cp6", "tp8",
                            "p7", "p5", "p3", "p1", "pz", "p2", "p4", "p6", "p8",
                            "po7", "po3", "poz", "po4", "po8",
                            "o1", "oz", "o2"],
        }

        token_dropout = []
        for mm in mmap:
            if random.random() < token_dropout_prob:
                n_ch = mm.shape[0]

                if chan_pos is None:
                    token_dropout.append([])
                    continue

                xyz_np = np.array(chan_pos, dtype=float)
                if np.abs(xyz_np).max() > 1.0:
                    xyz_np = xyz_np / 1000.0

                # Only allow montages with strictly fewer targets than n_ch —
                # mirrors the consumer-eeg guard above. Otherwise the
                # nearest-match degenerates to "keep all, drop nothing".
                eligible_montages = [
                    m_ for m_ in _STANDARD_MONTAGES_XYZ
                    if 0 < sum(1 for c in _STANDARD_MONTAGES_XYZ[m_] if c in _STD_XYZ) < n_ch
                ]
                if not eligible_montages:
                    token_dropout.append([])
                    continue

                montage_weights = [
                    sum(1 for c in _STANDARD_MONTAGES_XYZ[m_] if c in _STD_XYZ) ** 2
                    for m_ in eligible_montages
                ]

                iter_count = 0
                while True:

                    montage_name = random.choices(eligible_montages, weights=montage_weights, k=1)[0] # weight eligible montages with more chans more.
                    target_xyz = np.array(
                        [_STD_XYZ[ch] for ch in _STANDARD_MONTAGES_XYZ[montage_name] if ch in _STD_XYZ],
                        dtype=float,
                    )
                    # Keep the nearest actual channel to each target position
                    channels_to_keep = {
                        int(np.argmin(np.sqrt(((xyz_np - t) ** 2).sum(axis=1))))
                        for t in target_xyz
                    }

                    # check that all the montage target channels map to distinct data channels
                    if target_xyz.shape[0] == len(channels_to_keep):
                        channels_to_drop = sorted(set(range(n_ch)) - channels_to_keep)
                    else:
                        channels_to_drop = {}

                    iter_count += 1
                    if iter_count > 30:
                        channels_to_drop = []
                        break

                    if 3 < len(channels_to_drop) < n_ch - 3:
                        break

                N,T = mm.shape
                tc = T/num_fine_time_pts
                if tc%1 == 0:
                    tc_list = list(range(int(tc))) # list of coarse-time indices
                else:
                    print(f"Inside perform_token_dropout, Dropout scheme: {dropout_scheme}, Warning: {tc=} is not an integer!")

                combined_coords = [(r, t) for r in channels_to_drop for t in tc_list] # coords (chan, coarse-time) to drop
                token_dropout.append(combined_coords)

            else:
                token_dropout.append([]) # No dropout for this sample.


    elif dropout_scheme == "standard-montage-channel-dropout-old":
        # Standard 10-20/10-10 xyz positions (metres) used as target locations for each standard montage.
        # For each target position, the nearest actual channel is kept; all others are dropped.
        _STD_XYZ = {
            "fp1":  (-0.026,  0.083,  0.020), "fp2":  ( 0.026,  0.083,  0.020),
            "fpz":  ( 0.000,  0.087,  0.020),
            "af7":  (-0.068,  0.065,  0.015), "af8":  ( 0.068,  0.065,  0.015),
            "af3":  (-0.040,  0.071,  0.048), "af4":  ( 0.040,  0.071,  0.048),
            "afz":  ( 0.000,  0.073,  0.060),
            "f7":   (-0.083,  0.048,  0.012), "f8":   ( 0.083,  0.048,  0.012),
            "f5":   (-0.067,  0.050,  0.046), "f6":   ( 0.067,  0.050,  0.046),
            "f3":   (-0.047,  0.052,  0.063), "f4":   ( 0.047,  0.052,  0.063),
            "f1":   (-0.024,  0.054,  0.072), "f2":   ( 0.024,  0.054,  0.072),
            "fz":   ( 0.000,  0.054,  0.074),
            "ft7":  (-0.087,  0.025,  0.012), "ft8":  ( 0.087,  0.025,  0.012),
            "fc5":  (-0.073,  0.026,  0.052), "fc6":  ( 0.073,  0.026,  0.052),
            "fc3":  (-0.052,  0.026,  0.073), "fc4":  ( 0.052,  0.026,  0.073),
            "fc1":  (-0.026,  0.026,  0.085), "fc2":  ( 0.026,  0.026,  0.085),
            "fcz":  ( 0.000,  0.026,  0.087),
            "t7":   (-0.090,  0.000,  0.010), "t8":   ( 0.090,  0.000,  0.010),
            "c5":   (-0.078,  0.000,  0.046), "c6":   ( 0.078,  0.000,  0.046),
            "c3":   (-0.054,  0.000,  0.073), "c4":   ( 0.054,  0.000,  0.073),
            "c1":   (-0.027,  0.000,  0.087), "c2":   ( 0.027,  0.000,  0.087),
            "cz":   ( 0.000,  0.000,  0.090),
            "tp7":  (-0.087, -0.025,  0.012), "tp8":  ( 0.087, -0.025,  0.012),
            "cp5":  (-0.073, -0.026,  0.052), "cp6":  ( 0.073, -0.026,  0.052),
            "cp3":  (-0.052, -0.026,  0.073), "cp4":  ( 0.052, -0.026,  0.073),
            "cp1":  (-0.026, -0.026,  0.085), "cp2":  ( 0.026, -0.026,  0.085),
            "cpz":  ( 0.000, -0.026,  0.087),
            "p7":   (-0.083, -0.048,  0.012), "p8":   ( 0.083, -0.048,  0.012),
            "p5":   (-0.067, -0.050,  0.046), "p6":   ( 0.067, -0.050,  0.046),
            "p3":   (-0.047, -0.052,  0.063), "p4":   ( 0.047, -0.052,  0.063),
            "p1":   (-0.024, -0.054,  0.072), "p2":   ( 0.024, -0.054,  0.072),
            "pz":   ( 0.000, -0.054,  0.074),
            "po7":  (-0.068, -0.065,  0.015), "po8":  ( 0.068, -0.065,  0.015),
            "po3":  (-0.040, -0.071,  0.048), "po4":  ( 0.040, -0.071,  0.048),
            "poz":  ( 0.000, -0.073,  0.060),
            "o1":   (-0.026, -0.083,  0.020), "o2":   ( 0.026, -0.083,  0.020),
            "oz":   ( 0.000, -0.087,  0.020),
        }

        _STANDARD_MONTAGES_XYZ = {
            "standard_8":  ["fp1", "fp2", "c3", "cz", "c4", "o1", "o2", "pz"],
            "standard_16": ["fp1", "fp2",
                            "f3", "fz", "f4",
                            "c3", "cz", "c4",
                            "t7", "t8",
                            "p3", "pz", "p4",
                            "o1", "oz", "o2"],
            "standard_32": ["fp1", "fp2", "fpz",
                            "af3", "af4",
                            "f7", "f3", "fz", "f4", "f8",
                            "fc5", "fc1", "fcz", "fc2", "fc6",
                            "t7", "c3", "cz", "c4", "t8",
                            "cp5", "cp1", "cpz", "cp2", "cp6",
                            "p7", "p3", "pz", "p4", "p8",
                            "o1", "oz", "o2"],
            "standard_64": ["fp1", "fp2", "fpz",
                            "af7", "af3", "afz", "af4", "af8",
                            "f7", "f5", "f3", "f1", "fz", "f2", "f4", "f6", "f8",
                            "ft7", "fc5", "fc3", "fc1", "fcz", "fc2", "fc4", "fc6", "ft8",
                            "t7", "c5", "c3", "c1", "cz", "c2", "c4", "c6", "t8",
                            "tp7", "cp5", "cp3", "cp1", "cpz", "cp2", "cp4", "cp6", "tp8",
                            "p7", "p5", "p3", "p1", "pz", "p2", "p4", "p6", "p8",
                            "po7", "po3", "poz", "po4", "po8",
                            "o1", "oz", "o2"],
        }

        token_dropout = []
        for mm in mmap:
            if random.random() < token_dropout_prob:
                n_ch = mm.shape[0]

                if chan_pos is None:
                    token_dropout.append([])
                    continue

                xyz_np = np.array(chan_pos, dtype=float)
                if np.abs(xyz_np).max() > 1.0:
                    xyz_np = xyz_np / 1000.0

                iter_count = 0
                while True:
                    montage_name = random.choice(list(_STANDARD_MONTAGES_XYZ))
                    target_xyz = np.array(
                        [_STD_XYZ[ch] for ch in _STANDARD_MONTAGES_XYZ[montage_name] if ch in _STD_XYZ],
                        dtype=float,
                    )
                    # Keep the nearest actual channel to each target position
                    channels_to_keep = {
                        int(np.argmin(np.sqrt(((xyz_np - t) ** 2).sum(axis=1))))
                        for t in target_xyz
                    }

                    # check that all the montage target channels map to distinct data channels
                    if target_xyz.shape[0] == len(channels_to_keep):
                        channels_to_drop = sorted(set(range(n_ch)) - channels_to_keep)
                    else:
                        channels_to_drop = {}

                    iter_count += 1
                    if iter_count > 30:
                        channels_to_drop = []
                        break

                    if 3 < len(channels_to_drop) < n_ch - 3:
                        break

                N,T = mm.shape
                tc = T/num_fine_time_pts
                if tc%1 == 0:
                    tc_list = list(range(int(tc))) # list of coarse-time indices
                else:
                    print(f"Inside perform_token_dropout, Dropout scheme: {dropout_scheme}, Warning: {tc=} is not an integer!")

                combined_coords = [(r, t) for r in channels_to_drop for t in tc_list] # coords (chan, coarse-time) to drop
                token_dropout.append(combined_coords)

            else:
                token_dropout.append([]) # No dropout for this sample.


    elif dropout_scheme == "standard-montage-channel-dropout-by-name":
        # Dropout channels by standard montage name.
        #
        MONTAGE_CHANNELS = {
            "standard_8": {"fp1", "fp2", "c3", "cz", "c4", "o1", "o2", "pz"},
            "standard_16": {
                "fp1", "fp2",
                "f3", "fz", "f4",
                "c3", "cz", "c4",
                "t7", "t8",
                "p3", "pz", "p4",
                "o1", "oz", "o2",
            },
            "standard_19": {
                "fp1", "fp2",
                "f7", "f3", "fz", "f4", "f8",
                "t3", "c3", "cz", "c4", "t4",
                "t5", "p3", "pz", "p4", "t6",
                "o1", "o2",
            },
            "standard_32": {
                "fp1", "fp2", "fpz",
                "af3", "af4",
                "f7", "f3", "fz", "f4", "f8",
                "fc5", "fc1", "fcz", "fc2", "fc6",
                "t7", "c3", "cz", "c4", "t8",
                "cp5", "cp1", "cpz", "cp2", "cp6",
                "p7", "p3", "pz", "p4", "p8",
                "o1", "oz", "o2",
            },
            "standard_64": {
                "fp1", "fp2", "fpz",
                "af7", "af3", "afz", "af4", "af8",
                "f7", "f5", "f3", "f1", "fz", "f2", "f4", "f6", "f8",
                "ft7", "fc5", "fc3", "fc1", "fcz", "fc2", "fc4", "fc6", "ft8",
                "t7", "c5", "c3", "c1", "cz", "c2", "c4", "c6", "t8",
                "tp7", "cp5", "cp3", "cp1", "cpz", "cp2", "cp4", "cp6", "tp8",
                "p7", "p5", "p3", "p1", "pz", "p2", "p4", "p6", "p8",
                "po7", "po3", "poz", "po4", "po8",
                "o1", "oz", "o2",
            },
        }

        token_dropout = []
        for mm in mmap:
            if random.random() < token_dropout_prob:
                iter_count = 0
                while True:
                    headset = random.choice(list(MONTAGE_CHANNELS))
                    channels_to_keep = MONTAGE_CHANNELS[headset]
                    channels_to_drop = sorted([
                        i for i, name in enumerate(channel_names)
                        if name not in channels_to_keep
                    ])
                    iter_count += 1
                    if iter_count > 30:
                        channels_to_drop = []
                        break

                    # sample headset again if we don't have any channels to drop or we drop all channels
                    if 3 < len(channels_to_drop) < len(channel_names)-3:
                        break
                
                N,T = mm.shape
                tc = T/num_fine_time_pts
                if tc%1 == 0:
                    tc_list = list(range(int(tc))) # list of coarse-time indices
                else:
                    print(f"Inside perform_token_dropout, Dropout scheme: {dropout_scheme}, Warning: {tc=} is not an integer!")

                combined_coords = [(r, t) for r in channels_to_drop for t in tc_list] # coords (chan, coarse-time) to drop
                token_dropout.append(combined_coords)

            else:
                token_dropout.append([]) # No dropout for this sample.


    elif dropout_scheme == "consumer-eeg-channel-dropout":
        # Standard 10-20 xyz positions (metres) used as target locations for each headset.
        # For each target position, the nearest actual channel is kept; all others are dropped.
        _STD_XYZ = {
            "fp1":  (-0.026,  0.083,  0.020), "fp2":  ( 0.026,  0.083,  0.020),
            "af7":  (-0.068,  0.065,  0.015), "af8":  ( 0.068,  0.065,  0.015),
            "af3":  (-0.040,  0.071,  0.048), "af4":  ( 0.040,  0.071,  0.048),
            "f7":   (-0.083,  0.048,  0.012), "f8":   ( 0.083,  0.048,  0.012),
            "f5":   (-0.067,  0.050,  0.046), "f6":   ( 0.067,  0.050,  0.046),
            "f3":   (-0.047,  0.052,  0.063), "f4":   ( 0.047,  0.052,  0.063),
            "fz":   ( 0.000,  0.054,  0.074),
            "fc5":  (-0.073,  0.026,  0.052), "fc6":  ( 0.073,  0.026,  0.052),
            "fc1":  (-0.026,  0.026,  0.085), "fc2":  ( 0.026,  0.026,  0.085),
            "fcz":  ( 0.000,  0.026,  0.087),
            "t7":   (-0.090,  0.000,  0.010), "t8":   ( 0.090,  0.000,  0.010),
            "c3":   (-0.054,  0.000,  0.073), "c4":   ( 0.054,  0.000,  0.073),
            "cz":   ( 0.000,  0.000,  0.090),
            "tp9":  (-0.087, -0.032, -0.015), "tp10": ( 0.087, -0.032, -0.015),
            "cp5":  (-0.073, -0.026,  0.052), "cp6":  ( 0.073, -0.026,  0.052),
            "cp3":  (-0.052, -0.026,  0.073), "cp4":  ( 0.052, -0.026,  0.073),
            "cp1":  (-0.026, -0.026,  0.085), "cp2":  ( 0.026, -0.026,  0.085),
            "cpz":  ( 0.000, -0.026,  0.087),
            "p7":   (-0.083, -0.048,  0.012), "p8":   ( 0.083, -0.048,  0.012),
            "p3":   (-0.047, -0.052,  0.063), "p4":   ( 0.047, -0.052,  0.063),
            "pz":   ( 0.000, -0.054,  0.074),
            "po7":  (-0.068, -0.065,  0.015), "po8":  ( 0.068, -0.065,  0.015),
            "po3":  (-0.040, -0.071,  0.048), "po4":  ( 0.040, -0.071,  0.048),
            "o1":   (-0.026, -0.083,  0.020), "o2":   ( 0.026, -0.083,  0.020),
            "oz":   ( 0.000, -0.087,  0.020),
        }

        _CONSUMER_HEADSETS_XYZ = {
            "muse":           ["tp9", "af7", "af8", "tp10"],
            "crown":          ["cp3", "c3", "f5", "po3", "po4", "f6", "c4", "cp4"],
            "emotiv_epoc":    ["af3", "f7", "f3", "fc5", "t7", "p7", "o1",
                               "o2", "p8", "t8", "fc6", "f4", "f8", "af4"],
            "emotiv_insight": ["af3", "af4", "t7", "t8", "pz"],
            "unicorn":        ["fz", "c3", "cz", "c4", "pz", "po7", "oz", "po8"],
            "openbci_8":      ["fp1", "fp2", "c3", "c4", "p7", "p8", "o1", "o2"],
            "dreem":          ["fp1", "fp2", "o1", "o2", "cz"],
            "emotiv_flex32":  ["af3", "af4", "f7", "f3", "fz", "f4", "f8",
                               "fc5", "fc1", "fcz", "fc2", "fc6",
                               "t7", "c3", "cz", "c4", "t8",
                               "cp5", "cp1", "cpz", "cp2", "cp6",
                               "p7", "p3", "pz", "p4", "p8",
                               "po7", "po8", "o1", "oz", "o2"],
        }

        token_dropout = []
        for mm in mmap:
            if random.random() < token_dropout_prob:
                n_ch = mm.shape[0]

                if chan_pos is None:
                    token_dropout.append([])
                    continue

                xyz_np = np.array(chan_pos, dtype=float)
                if np.abs(xyz_np).max() > 1.0:
                    xyz_np = xyz_np / 1000.0

                iter_count = 0
                while True:
                    headset_name = random.choice(list(_CONSUMER_HEADSETS_XYZ))
                    target_xyz = np.array(
                        [_STD_XYZ[ch] for ch in _CONSUMER_HEADSETS_XYZ[headset_name] if ch in _STD_XYZ],
                        dtype=float,
                    )
                    # Keep the nearest actual channel to each target position
                    channels_to_keep = {
                        int(np.argmin(np.sqrt(((xyz_np - t) ** 2).sum(axis=1))))
                        for t in target_xyz
                    }

                    # check that all the headset target channels are in the data
                    if target_xyz.shape[0] == len(channels_to_keep):
                        channels_to_drop = sorted(set(range(n_ch)) - channels_to_keep)
                    else:
                        channels_to_drop = {}

                    iter_count += 1
                    if iter_count > 30:
                        channels_to_drop = []
                        break

                    if 3 < len(channels_to_drop) < n_ch - 3:
                        break

                N,T = mm.shape
                tc = T/num_fine_time_pts
                if tc%1 == 0:
                    tc_list = list(range(int(tc))) # list of coarse-time indices
                else:
                    print(f"Inside perform_token_dropout, Dropout scheme: {dropout_scheme}, Warning: {tc=} is not an integer!")

                combined_coords = [(r, t) for r in channels_to_drop for t in tc_list] # coords (chan, coarse-time) to drop
                token_dropout.append(combined_coords)

            else:
                token_dropout.append([]) # No dropout for this sample.



    elif dropout_scheme == "consumer-eeg-channel-dropout-by-name":
        #
        #
        CONSUMER_HEADSETS = {
            "muse":           {"tp9", "af7", "af8", "tp10"},
            "crown":          {"cp3", "c3", "f5", "po3", "po4", "f6", "c4", "cp4"},
            "emotiv_epoc":    {"af3", "f7", "f3", "fc5", "t7", "p7", "o1",
                               "o2", "p8", "t8", "fc6", "f4", "f8", "af4"},
            "emotiv_insight": {"af3", "af4", "t7", "t8", "pz"},
            "unicorn":        {"fz", "c3", "cz", "c4", "pz", "po7", "oz", "po8"},
            "openbci_8":      {"fp1", "fp2", "c3", "c4", "p7", "p8", "o1", "o2"},
            "dreem":          {"fp1", "fp2", "o1", "o2", "cz"},
            "emotiv_flex32":  {"af3", "af4", "f7", "f3", "fz", "f4", "f8",
                               "fc5", "fc1", "fcz", "fc2", "fc6",
                               "t7", "c3", "cz", "c4", "t8",
                               "cp5", "cp1", "cpz", "cp2", "cp6",
                               "p7", "p3", "pz", "p4", "p8",
                               "po7", "po8", "o1", "oz", "o2"},
        }

        token_dropout = []
        for mm in mmap:
            if random.random() < token_dropout_prob:
                iter_count = 0
                while True:
                    headset = random.choice(list(CONSUMER_HEADSETS))
                    channels_to_keep = CONSUMER_HEADSETS[headset]
                    channels_to_drop = sorted([
                        i for i, name in enumerate(channel_names)
                        if name not in channels_to_keep
                    ])
                    iter_count += 1
                    if iter_count > 30:
                        channels_to_drop = []
                        break

                    # sample headset again if we don't have any channels to drop or we drop all channels
                    if 3 < len(channels_to_drop) < len(channel_names)-3:
                        break

                N,T = mm.shape
                tc = T/num_fine_time_pts
                if tc%1 == 0:
                    tc_list = list(range(int(tc))) # list of coarse-time indices
                else:
                    print(f"Inside perform_token_dropout, Dropout scheme: {dropout_scheme}, Warning: {tc=} is not an integer!")

                combined_coords = [(r, t) for r in channels_to_drop for t in tc_list] # coords (chan, coarse-time) to drop
                token_dropout.append(combined_coords)

            else:
                token_dropout.append([]) # No dropout for this sample.


    elif dropout_scheme == "eval-1" or dropout_scheme == "full-channel-random-dropout-eval":
        ## NOTE: THIS FIXED DROPOUT RATE SCHEME USED FOR EVALS. FIRST, RANDOMLY DROP p*N CHANNELS.
        #        CAN ALSO DROP OUT CHANNELS IN AN ORGANIZED WAY FROM THE GRID.
        token_dropout = []
        for mm in mmap:
            N,T = mm.shape
            tc = T/num_fine_time_pts
            if tc%1 == 0:
                tc_list = list(range(int(tc))) # list of coarse-time indices
            else:
                print(f"Inside perform_token_dropout, Dropout scheme: {dropout_scheme}, Warning: {tc=} is not an integer!")

            if N<=1: # if there is only 1 channel, cannot dropout any.
                token_dropout.append([]) # No dropout for this sample.
                continue

            M = int(token_dropout_prob * N)
            random_integers = sorted(random.sample(range(1, N), M))
            combined_coords = [(r, t) for r in random_integers for t in tc_list] # coords (chan, coarse-time) to drop
            token_dropout.append(combined_coords)

    elif dropout_scheme == "no-dropout":
        token_dropout = [[]] * len(mmap)



    else:
        print(f"Dropout scheme: {dropout_scheme} not implemented - NOT DOING ANY DROPOUT!!")
        token_dropout = []
        
    return token_dropout


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #




class EEGDataset_v3(IterableDataset): #jm | loads from v7 mmap format (JSON sidecar + .dat files)
    """
    Iterable dataset loading from the v7 preprocessing mmap format.
    Mirrors EEGDataset_v2 output format exactly (packed_batch list of dicts).

    Key differences vs EEGDataset_v2:
      - Reads .dat memmaps + JSON sidecars instead of .pt zarr files.
      - Duration-weighted random file sampling instead of hard file sharding.
      - Variable window length per sample, drawn from V3_DURATION_RANGES.
      - Quality-based channel filtering at load time.
    """
    def __init__(self, args: BCIDatasetArgs):
        print(f"Inside EEGDataset_v3 with {args.data_dir=}")
        self.mmap_dir               = Path(args.data_dir)  #jm — reuses data_dir as mmap root
        self.filter_version         = args.filter_version
        self.min_quality_any        = args.min_quality_any   #jm
        self.min_quality_mean       = args.min_quality_mean  #jm
        self.dataset_id             = args.dataset_id
        self.shuffle                = args.shuffle
        self.seed                   = args.seed
        self.num_workers            = args.num_workers
        self._current_epoch         = 0
        self.num_fine_time_pts      = args.num_fine_time_pts
        self.sample_rate            = args.sample_rate
        self.use_coarse_time        = args.use_coarse_time
        self.cat_chan_xyz_and_eeg   = args.cat_chan_xyz_and_eeg
        self.target_packed_seqlen   = args.target_packed_seqlen
        self.pad_packed_seqlen      = args.pad_packed_seqlen   # CLODE
        self.token_dropout_prob     = args.token_dropout_prob
        self.dropout_scheme         = args.dropout_scheme
        self.num_bins               = args.num_bins_discretize_xyz_chan_pos
        self.stft_global_sigma      = args.stft_global_sigma
        self.sample_duration_str    = args.sample_duration_str
        self.do_avg_ref             = args.do_avg_ref
        self.z_score_type           = args.z_score_type
        self.mmap_sample_start      = args.mmap_sample_start
        self.mmap_sample_stop       = args.mmap_sample_stop
        self.skip_preepoched_data   = args.skip_preepoched_data


        #jm | Duration window sampling config for EEGDataset_v3.
        # Each entry: (min_sec, max_sec, relative_weight). Windows are snapped to tf-sample multiples.
        if args.sample_duration_str == "30_seconds":
            self.V3_DURATION_RANGES = [
                (0.5,  1.5,  0.20),   # very short   — low priority
                (1.5,  5.0,  0.30),   # 1–5 s        — highest priority
                (5.0, 10.0,  0.30),   # 5–10 s       — medium priority
                (10.0, 30.0, 0.20),   # >10 s        — lowest priority
            ]
        elif args.sample_duration_str == "10_seconds":
            self.V3_DURATION_RANGES = [
                (0.5,  1.0,  0.20),   # very short
                (1.0,  5.0,  0.60),   # 1–5  s   
                (5.0, 10.0,  0.20),   # 5–10 s     
            ]
        elif args.sample_duration_str == "5_seconds":
            self.V3_DURATION_RANGES = [
                (0.5,  5.0,  1.00),   # 0.5–5 s     
            ]
        elif args.sample_duration_str == "5_sec_wt_short_third":
            self.V3_DURATION_RANGES = [
                (0.5,  1.5,  0.333),
                (1.0,  5.0,  0.666),  
            ]
        elif args.sample_duration_str == "5_sec_wt_short_half":
            self.V3_DURATION_RANGES = [
                (0.5,  1.5,  0.5),
                (1.0,  5.0,  0.5),  
            ]
        elif args.sample_duration_str == "10_to_30_sec_half":
            self.V3_DURATION_RANGES = [
                (5.0, 10.0,  0.5),  
                (10.0, 30.0,  0.5),  
            ]
        elif args.sample_duration_str == "10_to_30_sec_third":
            self.V3_DURATION_RANGES = [
                (5.0, 10.0,  0.666),  
                (10.0, 30.0,  0.333),  
            ]
        elif args.sample_duration_str == "30_seconds_fifths":
            self.V3_DURATION_RANGES = [
                (0.5,  1.5,  0.20),   
                (1.5,  5.0,  0.40),   
                (5.0, 10.0,  0.20),   
                (10.0, 30.0, 0.20),
            ]
        else:
            raise ValueError(f"Invalid value for args.sample_duration_str: {args.sample_duration_str}")

        # xyz_extremes — same values and logic as EEGDataset_v2
        if args.chan_pos_xyz_extremes_type == "old":
            self.xyz_extremes = 1.10*torch.tensor([
                [-0.0861, -0.1124, -0.0680],
                [0.0858, 0.0849, 0.1002]
            ])
        elif args.chan_pos_xyz_extremes_type == "fifteens":
            self.xyz_extremes = torch.tensor([
                [-0.15, -0.15, -0.15],
                [ 0.15,  0.15,  0.15]
            ])
        elif args.chan_pos_xyz_extremes_type == "thirteens":
            self.xyz_extremes = torch.tensor([
                [-0.13, -0.13, -0.13],
                [ 0.13,  0.13,  0.13]
            ])
        elif args.chan_pos_xyz_extremes_type == "twelves":
            self.xyz_extremes = torch.tensor([
                [-0.12, -0.12, -0.12],
                [ 0.12,  0.12,  0.12]
            ])
        else:
            raise ValueError(f"Invalid value for args.chan_pos_xyz_extremes_type: {args.chan_pos_xyz_extremes_type}")


        #jm | Build file index from v7 metadata JSONs
        meta_dir = self.mmap_dir / "metadata"
        self.file_index = []


        # Gather up a list that is a union of all the glob patterns in args.glob_filter.
        seen = set()
        glob_paths = []
        for pat in args.glob_filter:
            for p in sorted(meta_dir.glob(pat)):
                if p not in seen:
                    seen.add(p)
                    glob_paths.append(p)
        

        for json_path in glob_paths:
            if json_path.name.startswith(".done"):
                continue

            try:
                with open(json_path) as f:
                    m = json.load(f)

                xyz = np.array(m["xyz"], dtype=np.float32)
                if np.all(xyz == 0):
                    continue  # skip recordings with no 3-D coordinates

                # Loop over each filter version and add to the file index.
                for filter_v in self.filter_version:    
                    self.file_index.append({
                        "base_name":         m["base_name"],
                        "n_channels":        int(m["n_channels"]),
                        "n_samples":         int(m["n_samples"]),
                        "duration_sec":      float(m["duration_sec"]),
                        "fs":                int(m["fs"]),
                        "is_epoched":        bool(m.get("is_epoched", False)),
                        "n_epochs":          int(m.get("n_epochs", 1)),
                        "samples_per_epoch": int(m.get("samples_per_epoch", m["n_samples"])),
                        "n_segments":        int(m["n_segments"]),
                        "quality_file":      m["quality_file"],
                        "dat_file":          m["data_files"][filter_v],
                        "xyz":               xyz,
                        "channel_names":     m["channel_names"],
                        "samples_per_seg":   int(round(float(m.get("quality_segment_sec", 1.0)) * int(m["fs"]))),  #jm
                    })
            except Exception as e:
                print(f"Warning: skipping {json_path.name}: {e}")
                continue

        # Flag to skip and not use pre-epoched data before it was implemented in the preprocessing pipeline.
        if self.skip_preepoched_data:
            self.file_index = [r for r in self.file_index if r["is_epoched"]==False]


        #jm | Duration-weighted sampling probabilities (longer files sampled more often)
        if self.mmap_sample_start is not None or self.mmap_sample_stop is not None:
            # Figure out the duration for each file between mmap_sample_start and mmap_sample_stop.
            start_samp = self.mmap_sample_start
            stop_samp = self.mmap_sample_stop
            durations = np.array([ min(stop_samp, r["n_samples"]) - max(start_samp, 0) for r in self.file_index], dtype=np.float64)
            self.file_weights = durations / durations.sum()
            self.total_samps  = int(durations.sum())
        else:
            durations = np.array([r["n_samples"] for r in self.file_index], dtype=np.float64)
            self.file_weights = durations / durations.sum()
            self.total_samps  = int(durations.sum())

        tokens = np.array([np.round(d/self.num_fine_time_pts)*r['n_channels'] for r,d in zip(self.file_index, durations)], dtype=np.float64)
        print(f"In EEGDataset_v3.__init__, {len(self.file_index)} recordings, {durations.sum()/(3600*self.sample_rate):.1f} hours total, {tokens.sum()} tokens")



    def __len__(self):
        return 10**10

    def set_epoch(self, epoch):
        self._current_epoch = epoch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers_per_rank = worker_info.num_workers if worker_info else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        global_worker_id = rank * num_workers_per_rank + worker_id

        #jm | Soft sharding: each worker uses an independent numpy RNG seeded by rank+worker+epoch.
        if self.seed is not None:
            worker_seed = int(self.seed + (1e3 * rank) + (1e6 * worker_id) + (1e9 * self._current_epoch))
        else:
            worker_seed = int(time.time() * 1000) % (2**31) + global_worker_id
        rng = np.random.default_rng(worker_seed)
        random.seed(worker_seed)

        # Init for sequence packing (same as EEGDataset_v2)
        seqlen_accum = 0
        packed_batch  = []

        while True:
            file_idx = -1  #jm — ensure defined even if exception fires before assignment
            try:
                #jm | 1. Sample a file proportional to its duration
                file_idx = int(rng.choice(len(self.file_index), p=self.file_weights))
                rec = self.file_index[file_idx]
                tf  = self.num_fine_time_pts

                #jm | 2. Sample a window duration from V3_DURATION_RANGES
                range_weights = np.array([w for _, _, w in self.V3_DURATION_RANGES])
                range_weights = range_weights / range_weights.sum()
                chosen_range  = int(rng.choice(len(self.V3_DURATION_RANGES), p=range_weights))
                lo, hi, _     = self.V3_DURATION_RANGES[chosen_range]
                win_sec       = float(rng.uniform(lo, hi))
                win_samples   = max(tf, int(round(win_sec * rec["fs"] / tf)) * tf)
                win_samples   = min(win_samples, rec["n_samples"])
                if win_samples < tf:
                    continue


                #jm | 3. Load window from mmap (continuous or epoched)
                dat_path = self.mmap_dir / rec["dat_file"]
                if rec["is_epoched"]:

                    # Convert mmap_sample_start and mmap_sample_stop mmap_epoch_start and mmap_epoch_stop (basically discretize it).
                    # and Limit extraction of sample from mmap file to a window between mmap_sample_start and mmap_sample_stop, if they are not None.
                    # Note: This is rough and not exact, but is good enough for our purposes.
                    if self.mmap_sample_start is not None and self.mmap_sample_stop is not None:
                        mmap_epoch_start = max(0, self.mmap_sample_start // rec["samples_per_epoch"])
                        mmap_epoch_stop = min(self.mmap_sample_stop // rec["samples_per_epoch"], rec["n_epochs"])
                    else:
                        mmap_epoch_start = 0
                        mmap_epoch_stop = rec["n_epochs"]

                    epoch_idx = int(rng.integers(mmap_epoch_start, mmap_epoch_stop))
                    mm = np.memmap(str(dat_path), dtype="float32", mode="r",
                                   shape=(rec["n_epochs"], rec["n_channels"], rec["samples_per_epoch"]))
                    data_np   = np.array(mm[epoch_idx])
                    del mm

                    # Make sure the number of time points is a multiple of num_fine_time_pts.
                    if data_np.shape[1]%self.num_fine_time_pts != 0:
                        data_np = data_np[:, :data_np.shape[1]//self.num_fine_time_pts*self.num_fine_time_pts]  # chop off the extra time points

                    win_samples = data_np.shape[1]

                else:
                    # Limit extraction of sample from mmap file to a window between mmap_sample_start and mmap_sample_stop, if they are not None.
                    if self.mmap_sample_start is not None:
                        bound_start = max(0, self.mmap_sample_start)
                    else:
                        bound_start = 0
                    if self.mmap_sample_stop is not None:
                        if self.mmap_sample_stop > rec["n_samples"]:
                            print(f"Warning: mmap_sample_stop is greater than the number of samples in the file {dat_path}. Setting mmap_sample_stop to {rec['n_samples']}.")
                        bound_stop = min(self.mmap_sample_stop, rec["n_samples"])
                    else:
                        bound_stop = rec["n_samples"]
                    
                    # Sample a start index for the window from the mmap file.
                    max_start = max(bound_start, bound_stop - win_samples)
                    n_steps   = (max_start - bound_start) // tf # in units of coarse-time chunks
                    start     = bound_start + int(rng.integers(0, n_steps + 1)) * tf if n_steps > 0 else bound_start
                    start     = min(start, max_start)


                    mm = np.memmap(str(dat_path), dtype="float32", mode="r",
                                   shape=(rec["n_channels"], rec["n_samples"]))
                    data_np = np.array(mm[:, start:start + win_samples])
                    del mm

                    # (CW) - Debugging: Print the extracted sample bounds and shape.
                    assert bound_start <= start <= start + win_samples <= bound_stop, f"Invalid sample bounds: {bound_start=} {start=} {start + win_samples=} {bound_stop=}"
                    assert data_np.shape[1] == win_samples, f"Invalid sample shape: {data_np.shape[1]=} {win_samples=}"

                # print(f"In EEGDataset_v3.__iter__, just before Quality-based channel filter...")

                #jm | 4. Quality-based channel filter — window-specific, two thresholds
                q_path = self.mmap_dir / rec["quality_file"]
                q_mm   = np.memmap(str(q_path), dtype="float32", mode="r",
                                   shape=(rec["n_channels"], rec["n_segments"]))
                # NOTE: Quality matrix for non-epoched data is shape (n_channels, n_segments). where each segment is 1 second long and for epoched data is shape (n_channels, n_epochs).
                
                seg_size = rec["samples_per_seg"]
                if rec["is_epoched"]:
                    q_window = np.array(q_mm[:, epoch_idx:epoch_idx + 1])
                else:
                    seg_s = start // seg_size
                    seg_e = min((start + win_samples + seg_size - 1) // seg_size, rec["n_segments"])
                    q_window = np.array(q_mm[:, seg_s:seg_e])
                del q_mm
                q_any  = q_window.min(axis=1)
                q_mean = q_window.mean(axis=1)
                good_ch = np.where((q_any >= self.min_quality_any) & (q_mean >= self.min_quality_mean))[0]
                if len(good_ch) < 3:
                    continue
                data_np  = data_np[good_ch]
                xyz_good = rec["xyz"][good_ch]                
                channel_names = [rec["channel_names"][int(i)] for i in good_ch]

                # Average reference data_np
                if self.do_avg_ref:
                    data_np = data_np - data_np.mean(axis=0)


                # Normalize signal to make STD = 1.0
                eps = 1e-6 # add epsilon to avoid division by zero std.
                if self.z_score_type == "across_channel":
                    data_np = (data_np - data_np.mean(axis=1)[:, None]) / (data_np.std(axis=1)[:, None] + eps)
                elif self.z_score_type == "across_sample":
                    data_np = (data_np - data_np.mean()) / (data_np.std() + eps)
                elif self.z_score_type == "none":
                    pass
                else:
                    raise ValueError(f"Invalid std_norm_type: {self.z_score_type}")





                # Skip entire sample if it contains NaN values.
                if np.isnan(data_np).any():
                    print(f"Warning: data_np contains NaN values in {dat_path}")
                    continue





                #jm | 5. Convert to torch; build channel positions (same as EEGDataset_v2)

                eeg_t              = torch.from_numpy(data_np).float()
                chan_pos           = torch.tensor(xyz_good, dtype=torch.float32)
                chan_pos_discrete  = discretize_chan_pos(chan_pos, self.xyz_extremes, self.num_bins)
                

                #jm | 6. Channel dropout (mirrors EEGDataset_v2 dropout schemes exactly)
                token_dropout = perform_token_dropout(dropout_scheme=self.dropout_scheme, 
                                                      token_dropout_prob=self.token_dropout_prob, 
                                                      num_fine_time_pts=self.num_fine_time_pts, 
                                                      mmap=[eeg_t],
                                                      channel_names=channel_names,
                                                      chan_pos=chan_pos)

                assert len(token_dropout)==1 



                #jm | 7. Reshape signals (same call signature as EEGDataset_v2)
                reshaped = chop_and_reshape_signals(eeg_t, chan_pos, chan_pos_discrete, tf, self.use_coarse_time)

                if self.cat_chan_xyz_and_eeg:
                    eeg_cat = torch.cat((reshaped[1], reshaped[0]), dim=1)
                else:
                    eeg_cat = reshaped[0]

                #jm | 8. Pack into packed_batch — yield when target_packed_seqlen is reached (mirrors EEGDataset_v2)
                seqlen_accum += reshaped[5]
                if seqlen_accum < self.target_packed_seqlen:
                    chan_id = reshaped[3]
                    t_coarse = reshaped[4]
                    dropout_bool = torch.zeros_like(chan_id, dtype=torch.bool)
                    for cd,td in token_dropout[0]:
                        dropout_bool[(chan_id==cd) & (t_coarse==td)] = True




                    packed_batch.append(
                        {"eeg_signal":         eeg_cat,
                         "chan_pos":           reshaped[1],
                         "chan_pos_discrete":  reshaped[2],
                         "chan_id":            reshaped[3],
                         "t_coarse":           reshaped[4],
                         "seq_lens":           reshaped[5],
                         "max_tc":             reshaped[4].max().item() + 1,
                         "token_dropout":      dropout_bool,
                         "pad_mask":           torch.ones(reshaped[5], 1, dtype=torch.float32),  # CLODE: 1=real
                         "ids":                file_idx,
                         "dataset_id":         self.dataset_id}
                    )
                else:
                    # Pack last truncated sample. And Yield packed batch.
                    seqlen_accum -= reshaped[5]
                    tokens_left   = self.target_packed_seqlen - seqlen_accum
                    if self.use_coarse_time == "A":
                        num_chans_r = reshaped[3].max().item() + 1
                        num_tc      = tokens_left // num_chans_r
                        tokens_left = num_chans_r * num_tc
                    elif self.use_coarse_time == "B":
                        num_tc      = reshaped[4].max().item() + 1
                        num_chans_r = tokens_left // num_tc
                        tokens_left = num_chans_r * num_tc
                    else:
                        raise ValueError(f"Unsupported use_coarse_time={self.use_coarse_time} for truncated sample in EEGDataset_v3")

                    if tokens_left > 0:
                        chan_id = reshaped[3][:tokens_left]
                        t_coarse = reshaped[4][:tokens_left]
                        dropout_bool = torch.zeros_like(chan_id, dtype=torch.bool)
                        for cd,td in token_dropout[0]:
                            dropout_bool[(chan_id==cd) & (t_coarse==td)] = True


                        packed_batch.append(
                            {"eeg_signal":        eeg_cat[:tokens_left],
                            "chan_pos":           reshaped[1][:tokens_left],
                            "chan_pos_discrete":  reshaped[2][:tokens_left],
                            "chan_id":            reshaped[3][:tokens_left],
                            "t_coarse":           reshaped[4][:tokens_left],
                            "seq_lens":           tokens_left,
                            "max_tc":             reshaped[4][:tokens_left].max().item() + 1,
                            "token_dropout":      dropout_bool,
                            "pad_mask":           torch.ones(tokens_left, 1, dtype=torch.float32),  # CLODE: 1=real
                            "ids":                file_idx,
                            "dataset_id":         self.dataset_id}
                        )

                    # CLODE: pad up to EXACTLY target_packed_seqlen as ONE extra all-zero
                    #        document. Becomes its own block in the doc mask (isolated in
                    #        attention) and is zeroed out of every loss via pad_mask.
                    cur_total = sum(item["seq_lens"] for item in packed_batch)
                    n_pad = self.target_packed_seqlen - cur_total
                    if self.pad_packed_seqlen and n_pad > 0:        # CLODE: gated by config flag
                        ref = packed_batch[0]
                        def _padz(key):
                            v = ref[key]
                            return torch.zeros((n_pad, *v.shape[1:]), dtype=v.dtype)
                        packed_batch.append({
                            "eeg_signal":        _padz("eeg_signal"),
                            "chan_pos":          _padz("chan_pos"),
                            "chan_pos_discrete": _padz("chan_pos_discrete"),
                            "chan_id":           _padz("chan_id"),
                            "t_coarse":          _padz("t_coarse"),
                            "seq_lens":          n_pad,                 # pad is its own document
                            "max_tc":            1,
                            "token_dropout":     _padz("token_dropout"),    # bool zeros (dtype carried)
                            "pad_mask":          torch.zeros(n_pad, 1, dtype=torch.float32),  # 0=pad
                            "ids":               -1,
                            "dataset_id":        ref["dataset_id"],
                        })

                    yield packed_batch
                    seqlen_accum = 0
                    packed_batch  = []

            except Exception as e:
                import traceback
                print(f"Error in EEGDataset_v3 (file_idx={file_idx}): {e}\n{traceback.format_exc()}")
                continue

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class EEGDataset_v2(IterableDataset):
    """
    Iterable dataset because we have lots more data for training.
    """
    def __init__(self, args: BCIDatasetArgs):

        print(f"Inside EEGDataset_v2 with {args.glob_filter=}")
        self.memmap_paths = list(Path(args.data_dir).glob(args.glob_filter))
        self.shuffle = args.shuffle
        self.seed = args.seed
        self.num_workers = args.num_workers 
        self.output_channels = args.decoder_input_channels
        self._current_epoch = 0 # To be updated by the training loop
        self.num_fine_time_pts = args.num_fine_time_pts
        self.sample_rate = args.sample_rate
        self.use_coarse_time = args.use_coarse_time
        self.cat_chan_xyz_and_eeg = args.cat_chan_xyz_and_eeg
        self.target_packed_seqlen = args.target_packed_seqlen
        self.do_N_epochs = args.do_N_epochs
        self.glob_filter = args.glob_filter
        self.chan_num_filter = args.chan_num_filter
        self.min_sample_duration = int(args.min_sample_duration_seconds * args.sample_rate)
        self.max_sample_duration = int(args.max_sample_duration_seconds * args.sample_rate)
        self.randomly_permute_sequence = args.randomly_permute_sequence
        self.token_dropout_prob = args.token_dropout_prob
        self.dropout_scheme = args.dropout_scheme
        self.num_bins = args.num_bins_discretize_xyz_chan_pos

        if args.chan_pos_xyz_extremes_type == "old":
            ## OLD TEST VALUES: (CW - WHAT I WAS USING PRIOR TO TEST104 and new v5 dataset)
            self.xyz_extremes = 1.10*torch.tensor([ 
                [-0.0861, -0.1124, -0.0680], 
                [0.0858, 0.0849, 0.1002]
            ])

        elif args.chan_pos_xyz_extremes_type == "fifteens":
            ## For new dataset with variable temporal length 
            self.xyz_extremes = torch.tensor([ 
                [-0.15, -0.15, -0.15], 
                [ 0.15,  0.15,  0.15]
            ])

        elif args.chan_pos_xyz_extremes_type == "thirteens":
            ##PICK WORKING VALUES BY EYE BALLING. (CW - USING THESE FOR TO TEST104 and new v5 dataset)
            self.xyz_extremes = torch.tensor([ 
                [-0.13, -0.13, -0.13], 
                [ 0.13,  0.13,  0.13]
            ])

        elif args.chan_pos_xyz_extremes_type == "twelves":
            ##PICK WORKING VALUES BY EYE BALLING. (CW - USING THESE FOR bigrun15 and new v5 dataset)
            self.xyz_extremes = torch.tensor([ 
                [-0.12, -0.12, -0.12], 
                [ 0.12,  0.12,  0.12]
            ])

        else:
            raise ValueError(f"Invalid value for args.chan_pos_xyz_extremes_type: {args.chan_pos_xyz_extremes_type} - must be one of 'old', 'thirteens'.")

        # Get total samps from all memmap files.
        print(f"Counting up total number of samples.")
        self.total_samps = 0
        for i, m_path in enumerate(self.memmap_paths):
            filename = os.path.basename(m_path).removesuffix('.pt')
            fparts =  filename.split('_')
            self.total_samps += int(fparts[-3])

        print(f"In Iterable EEGDataset.__init__, There are {len(self.memmap_paths)} memmap files")
        print(f"Total number of samples in one epoch of entire dataset is 🥁 🥁 🥁 : {self.total_samps}")

    def __len__(self):
        return self.total_samps

    def set_epoch(self, epoch):
        """
        Called by the main training loop to inform the dataset of the current epoch.
        NEED TO IMPLEMENT!
        """
        self._current_epoch = epoch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers_per_rank = worker_info.num_workers if worker_info else 1
        #
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        #
        global_worker_id = rank * num_workers_per_rank + worker_id
        total_global_workers = world_size * num_workers_per_rank
        
        if self.shuffle:
            # 1st. Set different deterministic random seeds for each rank and worker.    
            if self.seed is not None:
                base_seed = int(self.seed + (1e15 * self._current_epoch))
                rng_base = random.Random(base_seed)
                #
                worker_seed = int(self.seed + (1e3 * rank) \
                                            + (1e6 * worker_id) \
                                            + (1e15 * self._current_epoch))
                rng_worker = random.Random(worker_seed)
                torch.manual_seed(worker_seed)
                torch.cuda.manual_seed(worker_seed) 
                #
                g = torch.Generator()
                g.manual_seed(worker_seed)  
                #
                random.seed(worker_seed) # for shuffling list of samples
            else:
                g = None

            # 2nd. shuffle whole dataset files list with global seed (different for each epoch)
            rng_base.shuffle(self.memmap_paths) # in place shuffle of entire list of memmap files.

        # 3rd. Shard the indices of the memmap files across global workers. Each global worker processes a subset of memmap files. 
        sharded_indices_for_this_worker = list(
            range(global_worker_id, len(self.memmap_paths), total_global_workers)
        )

        if self.shuffle:    
            # 4th. Shuffle the indices assigned to this worker.\
            rng_worker.shuffle(sharded_indices_for_this_worker)


        # Init for sequence packing
        seqlen_accum = 0
        packed_batch = []

        # Loop over all the dataset files in this worker's shard.
        for ids in sharded_indices_for_this_worker:
            m_path = self.memmap_paths[int(ids)]
            mmap = torch.load(m_path, weights_only=False) #jm | this line was needed ONLY for the Moabb eval datasets (not sure why)


            # Handle different dataset structures
            if isinstance(mmap,dict):
                num_samps = len(mmap['data'])
                chan_pos = mmap['channel_positions']
                mmap = mmap['data']
            else: # assuming mmap is a tensor
                num_samps, num_chans, num_t = mmap.shape
                chan_pos = [torch.zeros(num_chans,3) for i in range(num_samps)]     # list of dummy channel positions (all-zeros).
                mmap = list(torch.unbind(mmap, dim=0))                              # turn 3D-tensor into list of tensors.


            # With variable length samples, now make sure each sample has some multiple of num_fine_time_pts time points.
            for i,m in enumerate(mmap):
                ch, tpts = m.shape
                if tpts%self.num_fine_time_pts!=0:
                    # chop off the extra time points
                    mmap[i] = m[:,:tpts//self.num_fine_time_pts*self.num_fine_time_pts]


            # Filter out samples that are less than min_sample_duration_seconds or greater than max_sample_duration_seconds.
            mmap_filt = []
            chan_pos_filt = []
            for i in range(len(mmap)):
                if mmap[i].shape[1] >= self.min_sample_duration and mmap[i].shape[1] <= self.max_sample_duration:
                    mmap_filt.append(mmap[i])
                    chan_pos_filt.append(chan_pos[i])
            mmap = mmap_filt
            chan_pos = chan_pos_filt

            chan_pos_discrete = [discretize_chan_pos(cp, self.xyz_extremes, self.num_bins) for cp in chan_pos]





            # Sanity check 3: 3D scatter plot of channel positions and discretized positions
            plot_chan_pos_comparison = False
            if plot_chan_pos_comparison:
                fig = plt.figure(figsize=(16, 7))

                # Left plot: Original continuous positions
                ax1 = fig.add_subplot(121, projection='3d')
                cp = chan_pos[0].cpu().numpy()
                ax1.scatter(cp[:, 0], cp[:, 1], cp[:, 2], c='blue', marker='o', s=50, alpha=0.4)
                for i in range(cp.shape[0]):
                    ax1.text(cp[i, 0], cp[i, 1], cp[i, 2], str(i), fontsize=8)
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_zlabel('Z')
                ax1.set_title('Original Channel Positions')

                # Right plot: Discretized positions
                ax2 = fig.add_subplot(122, projection='3d')
                cpd = chan_pos_discrete[0].cpu().numpy()
                ax2.scatter(cpd[:, 0], cpd[:, 1], cpd[:, 2], c='red', marker='s', s=50, alpha=0.4)
                for i in range(cpd.shape[0]):
                    ax2.text(cpd[i, 0], cpd[i, 1], cpd[i, 2], str(i), fontsize=8)
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                ax2.set_zlabel('Z')
                ax2.set_title('Discretized Channel Positions')

                plt.tight_layout()
                plt.savefig('figures/chan_pos_comparison.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved channel position comparison plot to figures/chan_pos_comparison.png")

            # Filter out samples that do not have self.chan_num_filter channels. This is pretty quick - not the source of data_t slowdown
            if self.chan_num_filter is not None:
                mmap_filt = []
                chan_pos_filt = []
                chan_pos_discrete_filt = []
                for i in range(len(mmap)):
                    if mmap[i].shape[0]==self.chan_num_filter:
                        mmap_filt.append(mmap[i])
                        chan_pos_filt.append(chan_pos[i])
                        chan_pos_discrete_filt.append(chan_pos_discrete[i])
                mmap = mmap_filt
                chan_pos = chan_pos_filt
                chan_pos_discrete = chan_pos_discrete_filt


            # Shuffle the channels randomly within data matrix to see if the model can still learn from concat'd {x,y,z}-position or RoPE on discretized xyz positions
            # Note: This is before things are reshaped into coarse-time and fine-time inside chop_and_reshape_signals()
            if self.randomly_permute_sequence:
                mmap_shuf = []
                chan_pos_shuf = []
                chan_pos_discrete_shuf = []
                for i in range(len(mmap)):
                    num_chans = mmap[i].shape[0]
                    shuffled_indices = torch.randperm(num_chans)
                    mmap_shuf.append(mmap[i][shuffled_indices])
                    chan_pos_shuf.append(chan_pos[i][shuffled_indices])
                    chan_pos_discrete_shuf.append(chan_pos_discrete[i][shuffled_indices])
                mmap = mmap_shuf
                chan_pos = chan_pos_shuf
                chan_pos_discrete = chan_pos_discrete_shuf





            token_dropout = perform_token_dropout(dropout_scheme=self.dropout_scheme, 
                                                  token_dropout_prob=self.token_dropout_prob, 
                                                  num_fine_time_pts=self.num_fine_time_pts, 
                                                  mmap=mmap)



            # 5th. Shuffle samples within mmap/chan_pos lists.
            # NOTE: Shuffle index before reshaping signals so I can compare before and after (out in eeg_eval.py) plots.
            #       Testing chop_and_reshape_signals() and invert_reshape_signals() functions with real signals.
            indx = list(range(len(mmap)))
            if self.shuffle:
                random.shuffle(indx)

            check_reshape_plots = False # Plot signals before and after reshaping to verify its working.
                                         # THIS IS NOT EXPECTED TO WORK WITH self.use_coarse_time=="D
            if check_reshape_plots:
                # Create a sample signal to demonstrate reshape and unreshape is working.
                tf = self.num_fine_time_pts
                tc = 10
                indx0 = indx[0]
                num_chans = mmap[indx0].shape[0]
                for i in range(num_chans):
                    signal = mmap[indx0][i,:]
                    if self.use_coarse_time=="C": # plot only the first tf part of signal it "C"
                        signal = signal[:tf]
                    fig, ax = plt.subplots(1, 1, figsize=(20, 4))
                    ax.plot(signal)
                    if self.use_coarse_time!="C": 
                        ax.scatter(tf*np.arange(tc), signal[::tf], color='red')
                    plt.savefig(f"figures/inspect_reshape_and_invert/test0_ch{i}_before.png", dpi=300, bbox_inches='tight')
                    plt.close()

            if self.use_coarse_time=="A" or self.use_coarse_time=="B" or self.use_coarse_time=="C" or self.use_coarse_time=="D":
                reshaped = [chop_and_reshape_signals(m, c, cd, self.num_fine_time_pts, self.use_coarse_time) for m,c,cd in zip(mmap, chan_pos, chan_pos_discrete)]
            else:
                print(f"Dont understand {self.use_coarse_time=}")



            # REFACTOR THIS: Flatten list of lists into single list if trying to process each channel as separate sample.
            if self.use_coarse_time=="D":
                r0 = []
                r1 = []
                r2 = []
                r3 = []
                r4 = []
                r5 = []
                for r in reshaped:
                    r0.extend( r[0] ) # eeg signal
                    r1.extend( r[1] ) # chan position
                    r2.extend( r[2] ) # discete chan position
                    r3.extend( r[3] ) # chan id
                    r4.extend( r[4] ) # t_coarse
                    r5.extend( r[5] ) # seq_len

                reshaped = []
                for i in range(len(r0)):
                    reshaped.append( (r0[i], r1[i], r2[i], r3[i], r4[i], r5[i]) )

            if self.cat_chan_xyz_and_eeg:
                eeg_cat = [torch.cat((res[1],res[0]),dim=1) for res in reshaped] # make eeg_signal = [{x,y,z}, (tf)]
            else:
                eeg_cat = [res[0] for res in reshaped]                           # make eeg_signal = [just (tf)]]

            # Inside EEGDataset_v2, what is shape of eeg_cat when cat_chan_xyz_and_eeg is True vs False?)
            # self.cat_chan_xyz_and_eeg=False --> eeg_cat[indx0].shape=torch.Size([210, 128])
            # self.cat_chan_xyz_and_eeg=True, --> eeg_cat[indx0].shape=torch.Size([210, 131])

            if check_reshape_plots:
                if self.use_coarse_time=="C":
                    tc=1
                num_chans = eeg_cat[indx0].shape[0]//tc
                if self.cat_chan_xyz_and_eeg:
                    xxx, _, _, _, _ = invert_reshape_signals(sig_reshaped=eeg_cat[indx0][:,3:],
                                                          pos_reshaped=reshaped[indx0][1],
                                                          num_chans=num_chans, 
                                                          tf=tf,
                                                          tc=reshaped[i][4].max().item()+1,
                                                          use_coarse_time=self.use_coarse_time,
                    )
                else:
                    xxx, _, _, _, _ = invert_reshape_signals(sig_reshaped=eeg_cat[indx0], 
                                                          pos_reshaped=reshaped[indx0][1],
                                                          num_chans=num_chans, 
                                                          tf=tf,
                                                          tc=reshaped[i][4].max().item()+1,
                                                          use_coarse_time=self.use_coarse_time,
                    )

                # Create a sample signal to demonstrate reshape and unreshape is working.
                for i in range(num_chans):
                    signal = xxx[i,:]
                    fig, ax = plt.subplots(1, 1, figsize=(20, 4))
                    ax.plot(signal)
                    ax.scatter(tf*np.arange(tc), signal[::tf], color='red')
                    plt.savefig(f"figures/inspect_reshape_and_invert/test0_ch{i}_after.png", dpi=300, bbox_inches='tight')
                    plt.close()  

            dataset_id = int(m_path.name.split('_')[0].removeprefix('ds'))    # standardized dataset id 🎉

            for s in indx:
                try:
                    # Collect up full samples in packed_batch until seqlen_accum > self.target_seqlen
                    seqlen_accum += reshaped[s][5]
                    if seqlen_accum < self.target_packed_seqlen:
                        
                        # Apply channel dropout here to get boolean mask
                        chan_id = reshaped[s][3]
                        t_coarse = reshaped[s][4]
                        tok_do = token_dropout[s]

                        # Create boolean mask to drop out the specified channels and time-points.
                        dropout_bool = torch.zeros_like(chan_id, dtype=torch.bool)
                        for cd,td in tok_do:
                            dropout_bool[(chan_id==cd) & (t_coarse==td)] = True


                        packed_batch.append(
                            {"eeg_signal": eeg_cat[s], 
                            "chan_pos": reshaped[s][1], 
                            "chan_pos_discrete": reshaped[s][2], 
                            "chan_id": reshaped[s][3],
                            "t_coarse":reshaped[s][4], 
                            "seq_lens":reshaped[s][5],  
                            "max_tc": reshaped[s][4].max().item()+1,
                            "token_dropout": dropout_bool,
                            "ids": ids, 
                            "dataset_id": dataset_id}
                        )
                    # Collect up partial sample to reach self.target_seqlen    
                    else:
                        seqlen_accum -= reshaped[s][5]                          # take off last sample's seq_len
                        tokens_left = self.target_packed_seqlen - seqlen_accum  # compute number of tokens left to fill

                        if self.use_coarse_time=="A":
                            # take as many tokens as we can up to tokens_left grabbing as many time-points for which we can have every channel.
                            num_chans = reshaped[s][3].max().item()+1
                            num_tc =  tokens_left // num_chans
                            tokens_left = num_chans * num_tc
                        elif self.use_coarse_time=="B":
                            # take as many tokens as we can up to tokens_left grabbing as many channels for which we can have every time-point.
                            num_tc = reshaped[s][4].max().item()+1
                            num_chans =  tokens_left // num_tc
                            tokens_left = num_chans * num_tc
                        else:
                            raise ValueError(f"I dont know what to do with last truncated sample in EEGDataset_v2 with self.use_coarse_time: {self.use_coarse_time}")
                        # Apply channel dropout here to get boolean mask
                        chan_id = reshaped[s][3][:tokens_left]
                        tok_do = token_dropout[s]
                        dropout_bool = torch.zeros_like(chan_id, dtype=torch.bool)
                        for cd,td in tok_do:
                            dropout_bool[(chan_id==cd) & (t_coarse==td)] = True

                        packed_batch.append(
                            {"eeg_signal": eeg_cat[s][:tokens_left], 
                            "chan_pos": reshaped[s][1][:tokens_left], 
                            "chan_pos_discrete": reshaped[s][2][:tokens_left], 
                            "chan_id": reshaped[s][3][:tokens_left],
                            "t_coarse":reshaped[s][4][:tokens_left], 
                            "seq_lens":tokens_left,  
                            "max_tc": reshaped[s][4][:tokens_left].max().item()+1,
                            "token_dropout": dropout_bool,
                            "ids": ids, 
                            "dataset_id": dataset_id}
                        )


                        # Then yield packed_batch and reset list to []
                        yield packed_batch
                        seqlen_accum = 0
                        packed_batch = []

                except Exception as e:
                    print(f"Error processing sample: {e} : {ids} : {m_path}")
                    continue


class EEGDataset_b2(IterableDataset):
    """

    NOTE: THIS IS BECOMING DEPRECATED. USE EEGDataset_v2 INSTEAD. BEREN SAID WE CAN JUST STREAM DATASET LOCALLY.
    Iterable dataset that pulls .pt files from Backblaze B2 bucket using boto3 S3-compatible API.
    Modeled after EEGDataset_v2 but with cloud storage integration.
    """
    def __init__(self, args: BCIDatasetArgs):
        print(f"Inside EEGDataset_b2 with B2 bucket: {args.b2_bucket_name}, prefix: {args.data_dir}")
        
        # Validate B2 configuration
        if not all([args.b2_bucket_name, args.b2_endpoint_url, args.b2_access_key_id, args.b2_secret_access_key]):
            raise ValueError("B2 configuration incomplete. Must provide: b2_bucket_name, b2_endpoint_url, b2_access_key_id, b2_secret_access_key")
        
        # Initialize boto3 S3 client for B2
        self.s3_client = boto3.client(
            's3',
            endpoint_url=args.b2_endpoint_url,
            aws_access_key_id=args.b2_access_key_id,
            aws_secret_access_key=args.b2_secret_access_key
        )
    
        self.bucket_name = args.b2_bucket_name
        self.key_prefix = args.data_dir or ""
        self.cache_dir = args.b2_local_cache_dir
        self.cache_files = args.b2_cache_files
        
        # Set up cache directory if caching is enabled
        if self.cache_files and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Store all other args (same as EEGDataset_v2)
        self.shuffle = args.shuffle
        self.seed = args.seed
        self.num_workers = args.num_workers
        self.output_channels = args.decoder_input_channels
        self._current_epoch = 0
        self.num_fine_time_pts = args.num_fine_time_pts
        self.use_coarse_time = args.use_coarse_time
        self.cat_chan_xyz_and_eeg = args.cat_chan_xyz_and_eeg
        self.target_packed_seqlen = args.target_packed_seqlen
        self.do_N_epochs = args.do_N_epochs
        self.glob_filter = args.glob_filter  # Used to filter keys (e.g., "**/*.pt")
        self.chan_num_filter = args.chan_num_filter
        self.min_sample_duration = int(args.min_sample_duration_seconds * args.sample_rate)
        self.max_sample_duration = int(args.max_sample_duration_seconds * args.sample_rate)
        self.randomly_permute_sequence = args.randomly_permute_sequence
        self.token_dropout_prob = args.token_dropout_prob
        self.dropout_scheme = args.dropout_scheme
        self.num_bins = args.num_bins_discretize_xyz_chan_pos

        if args.chan_pos_xyz_extremes_type == "old":
            ## OLD TEST VALUES: (CW - WHAT I WAS USING PRIOR TO TEST104 and new v5 dataset)
            self.xyz_extremes = 1.10*torch.tensor([ 
                [-0.0861, -0.1124, -0.0680], 
                [0.0858, 0.0849, 0.1002]
            ])
        elif args.chan_pos_xyz_extremes_type == "fifteens":
            self.xyz_extremes = torch.tensor([ 
                [-0.15, -0.15, -0.15], 
                [ 0.15,  0.15,  0.15]
            ])
        elif args.chan_pos_xyz_extremes_type == "thirteens":
            ##PICK WORKING VALUES BY EYE BALLING. (CW - USING THESE FOR TO TEST104 and new v5 dataset)
            self.xyz_extremes = torch.tensor([ 
                [-0.13, -0.13, -0.13], 
                [ 0.13,  0.13,  0.13]
            ])
        elif args.chan_pos_xyz_extremes_type == "twelves":
            self.xyz_extremes = torch.tensor([ 
                [-0.12, -0.12, -0.12], 
                [ 0.12,  0.12,  0.12]
            ])

        else:
            raise ValueError(f"Invalid value for args.chan_pos_xyz_extremes_type: {args.chan_pos_xyz_extremes_type} - must be one of 'old', 'thirteens', 'twelves'.")
        
        # List all .pt files in the B2 bucket/prefix
        print(f"Listing .pt files in B2 bucket: {self.bucket_name}, prefix: {self.key_prefix}.  Will take a few mins...")        
        

        self.b2_file_keys = self._list_b2_files()
        print(f"Found {len(self.b2_file_keys)} .pt files in B2 bucket")
        
        # Get total samps from all files (same logic as EEGDataset_v2)
        print(f"Counting up total number of samples.")
        self.total_samps = 0
        for key in self.b2_file_keys:
            filename = os.path.basename(key).removesuffix('.pt')
            fparts = filename.split('_')
            if len(fparts) >= 3:
                self.total_samps += int(fparts[-3])
        
        print(f"In Iterable EEGDataset_b2.__init__, There are {len(self.b2_file_keys)} B2 files")
        print(f"Total number of samples in one epoch of entire dataset is 🥁 🥁 🥁 : {self.total_samps}")
    
    def _list_b2_files(self):
        """List all .pt files in the B2 bucket with the given prefix."""
        
        file_keys = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.key_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.pt'):
                        # Apply glob filter if specified using fnmatch (simple pattern matching)
                        if fnmatch.fnmatch(key, self.glob_filter):
                            file_keys.append(key)
        
        return sorted(file_keys)
    
    def _get_cached_path(self, key: str) -> Optional[str]:
        """Get local cache path for a B2 key."""
        if not self.cache_dir:
            return None
        # Create safe filename from key
        safe_filename = key.replace('/', '_').replace('\\', '_')
        return os.path.join(self.cache_dir, safe_filename)
    
    def _download_file(self, key: str) -> str:
        """Download a file from B2 and return local path."""
        # Check cache first
        if self.cache_files and self.cache_dir:
            cached_path = self._get_cached_path(key)
            if cached_path and os.path.exists(cached_path):
                return cached_path
        
        # Download file
        if self.cache_files and self.cache_dir:
            local_path = self._get_cached_path(key)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
        else:
            # Use temp file if not caching
            fd, local_path = tempfile.mkstemp(suffix='.pt')
            os.close(fd)
        
        try:
            self.s3_client.download_file(self.bucket_name, key, local_path)
            return local_path
        except Exception as e:
            if not self.cache_files:
                # Clean up temp file on error
                if os.path.exists(local_path):
                    os.remove(local_path)
            raise e
    
    def _load_from_b2(self, key: str):
        """Download and load a .pt file from B2."""
        local_path = self._download_file(key)
        try:
            data = torch.load(local_path, map_location='cpu')
            return data
        finally:
            # Clean up temp file if not caching
            if not self.cache_files and os.path.exists(local_path):
                os.remove(local_path)
    
    def __len__(self):
        return self.total_samps
    
    def set_epoch(self, epoch):
        """Called by the main training loop to inform the dataset of the current epoch."""
        self._current_epoch = epoch
    
    def __iter__(self):
        # Same worker/distributed setup as EEGDataset_v2
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers_per_rank = worker_info.num_workers if worker_info else 1
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        global_worker_id = rank * num_workers_per_rank + worker_id
        total_global_workers = world_size * num_workers_per_rank
        
        if self.shuffle:
            if self.seed is not None:
                base_seed = int(self.seed + (1e15 * self._current_epoch))
                rng_base = random.Random(base_seed)
                
                worker_seed = int(self.seed + (1e3 * rank) 
                                            + (1e6 * worker_id) 
                                            + (1e15 * self._current_epoch))
                rng_worker = random.Random(worker_seed)
                torch.manual_seed(worker_seed)
                torch.cuda.manual_seed(worker_seed)
                
                g = torch.Generator()
                g.manual_seed(worker_seed)
                
                random.seed(worker_seed)
            else:
                g = None
                rng_base = random.Random()
                rng_worker = random.Random()
            
            # Shuffle file keys
            file_keys_copy = self.b2_file_keys.copy()
            rng_base.shuffle(file_keys_copy)
        else:
            file_keys_copy = self.b2_file_keys.copy()
        
        # Shard file keys across global workers
        sharded_indices_for_this_worker = list(
            range(global_worker_id, len(file_keys_copy), total_global_workers)
        )
        
        if self.shuffle:
            if self.seed is not None:
                rng_worker.shuffle(sharded_indices_for_this_worker)
            else:
                random.shuffle(sharded_indices_for_this_worker)
        
        # Init for sequence packing
        seqlen_accum = 0
        packed_batch = []


        # Loop over all the B2 files in this worker's shard
        for ids in sharded_indices_for_this_worker:
            b2_key = file_keys_copy[int(ids)]
            
            # Download and load from B2
            mmap = self._load_from_b2(b2_key)
            
            # Handle different dataset structures (same as EEGDataset_v2)
            if isinstance(mmap, dict):
                num_samps = len(mmap['data'])
                chan_pos = mmap['channel_positions']
                mmap = mmap['data']
            else:  # assuming mmap is a tensor
                num_samps, num_chans, num_t = mmap.shape
                chan_pos = [torch.zeros(num_chans, 3) for i in range(num_samps)]
                mmap = list(torch.unbind(mmap, dim=0))


            # With variable length samples, now make sure each sample has some multiple of num_fine_time_pts time points.
            for i,m in enumerate(mmap):
                ch, tpts = m.shape
                if tpts%self.num_fine_time_pts!=0:
                    # chop off the extra time points
                    mmap[i] = m[:,:tpts//self.num_fine_time_pts*self.num_fine_time_pts]


            # Filter out samples that are less than min_sample_duration or greater than max_sample_duration.
            mmap_filt = []
            chan_pos_filt = []
            for i in range(len(mmap)):
                if mmap[i].shape[1] >= self.min_sample_duration and mmap[i].shape[1] <= self.max_sample_duration:
                    mmap_filt.append(mmap[i])
                    chan_pos_filt.append(chan_pos[i])
            mmap = mmap_filt
            chan_pos = chan_pos_filt  

            # Discretize chan_pos
            chan_pos_discrete = [discretize_chan_pos(cp, self.xyz_extremes, self.num_bins) for cp in chan_pos]
            
            # Filter by channel number if specified
            if self.chan_num_filter is not None:
                mmap_filt = []
                chan_pos_filt = []
                chan_pos_discrete_filt = []
                for i in range(len(mmap)):
                    if mmap[i].shape[0] == self.chan_num_filter:
                        mmap_filt.append(mmap[i])
                        chan_pos_filt.append(chan_pos[i])
                        chan_pos_discrete_filt.append(chan_pos_discrete[i])
                mmap = mmap_filt
                chan_pos = chan_pos_filt
                chan_pos_discrete = chan_pos_discrete_filt
            
            # Randomly permute channels within data matrix
            if self.randomly_permute_sequence:
                mmap_shuf = []
                chan_pos_shuf = []
                chan_pos_discrete_shuf = []
                for i in range(len(mmap)):
                    num_chans = mmap[i].shape[0]
                    shuffled_indices = torch.randperm(num_chans)
                    mmap_shuf.append(mmap[i][shuffled_indices])
                    chan_pos_shuf.append(chan_pos[i][shuffled_indices])
                    chan_pos_discrete_shuf.append(chan_pos_discrete[i][shuffled_indices])
                mmap = mmap_shuf
                chan_pos = chan_pos_shuf
                chan_pos_discrete = chan_pos_discrete_shuf


            token_dropout = perform_token_dropout(dropout_scheme=self.dropout_scheme, 
                                                    token_dropout_prob=self.token_dropout_prob, 
                                                    num_fine_time_pts=self.num_fine_time_pts, 
                                                    mmap=mmap)
            
            
            # Shuffle samples within file
            indx = list(range(len(mmap)))
            if self.shuffle:
                random.shuffle(indx)
            
            # Reshape signals
            if self.use_coarse_time in {"A", "B", "C", "D"}:
                reshaped = [chop_and_reshape_signals(m, c, cd, self.num_fine_time_pts, self.use_coarse_time) 
                            for m, c, cd in zip(mmap, chan_pos, chan_pos_discrete)]
                
            else:
                print(f"Dont understand {self.use_coarse_time=}")
                continue
            
            # Flatten if use_coarse_time=="D"
            if self.use_coarse_time == "D":
                r0, r1, r2, r3, r4, r5 = [], [], [], [], [], []
                for r in reshaped:
                    r0.extend(r[0])
                    r1.extend(r[1])
                    r2.extend(r[2])
                    r3.extend(r[3])
                    r4.extend(r[4])
                    r5.extend(r[5])
                reshaped = []
                for i in range(len(r0)):
                    reshaped.append((r0[i], r1[i], r2[i], r3[i], r4[i], r5[i]))
            
            # Concatenate channel positions if enabled
            if self.cat_chan_xyz_and_eeg:
                eeg_cat = [torch.cat((res[1], res[0]), dim=1) for res in reshaped]
            else:
                eeg_cat = [res[0] for res in reshaped]
            
            # Extract dataset ID from filename
            filename = os.path.basename(b2_key)
            dataset_id = int(filename.split('_')[0].removeprefix('ds')) if filename.startswith('ds') else 0

            
            # Yield packed batches
            for s in indx:
                try:
                    # Collect up samples in packed_batch until seqlen_accum > self.target_packed_seqlen
                    seqlen_accum += reshaped[s][5]
                    if seqlen_accum < self.target_packed_seqlen:
                        
                        # Apply channel dropout boolean mask
                        chan_id = reshaped[s][3]
                        tok_do = token_dropout[s]
                        dropout_bool = torch.zeros_like(chan_id, dtype=torch.bool)
                        for d in tok_do:
                            dropout_bool[chan_id == d] = True
                        
                        packed_batch.append({
                            "eeg_signal": eeg_cat[s],
                            "chan_pos": reshaped[s][1],
                            "chan_pos_discrete": reshaped[s][2],
                            "chan_id": reshaped[s][3],
                            "t_coarse": reshaped[s][4],
                            "seq_lens": reshaped[s][5],
                            "max_tc": reshaped[s][4].max().item()+1,
                            "token_dropout": dropout_bool,
                            "ids": ids,
                            "dataset_id": dataset_id
                        })
                    else:


                        # NOTE: Would have to add truncated sample


                        yield packed_batch
                        seqlen_accum = 0
                        packed_batch = []
                
                except Exception as e:
                    print(f"Error processing sample: {e} : {ids} : {b2_key}")
                    continue


def beta_sched(t_shape, device, dtype):
    """
    Note: beta weights high and low noise values more! 
    This makes sense for audio, (maybe??) not for EEG
    """
    t = torch.randn(t_shape, device=device, dtype=dtype) * 2 + 0.3
    t = torch.sigmoid_(t) * 1.02 - 0.01
    return t.clamp_(0,1)

def logit_normal_sched(t_shape, device, dtype, m=0.0, s=1.0):
    """Logit-normal time sampler:  t = sigmoid(m + s*z), z~N(0,1).

    m=0, s=1 (defaults) gives a single hump centred at t=0.5 (SD3-style),
    as opposed to beta_sched's U-shape (s=2 -> bimodal, mass at the edges).
    Output is strictly in (0,1)

    If you want the hump sharper, drop s toward 0.6-0.8. If you want it nudged
    toward higher-noise t (often helps the harder denoising end), bump m to
    +0.2..+0.5.

    """
    z = torch.randn(t_shape, device=device, dtype=dtype)
    return torch.sigmoid(m + s * z)


class EEGProcessor:
    def __init__(self, args: BCIDatasetArgs):
        self.diffusion_noise_schedule = args.diffusion_noise_schedule
        self.logit_normal_mean = args.logit_normal_mean   
        self.logit_normal_std  = args.logit_normal_std    

        self.global_sigma = args.stft_global_sigma
        self.patch_type = args.patching_type
        self.diffusion_forcing = args.diffusion_forcing
        self.cat_chan_xyz_and_eeg = args.cat_chan_xyz_and_eeg
        self.dont_noise_chan_xyz = args.dont_noise_chan_xyz
        self.masked_in_decoder = args.masked_in_decoder
        if self.diffusion_forcing:
            self.diffusion_forcing_num_frames = args.diffusion_forcing_num_frames


    def to(self, device):
        return self # (CW)
        # Unlike STFTProcessor in AY2latent/data_lean.py, nothing to put on device



    @torch.compile() # (CW) - REINSTATE: commented out for now while working with dropout_chans
    def process(self, eeg_signal, chan_pos, chan_pos_discrete, chan_id, t_coarse, seq_lens, max_tc, token_dropout, pad_mask=None): # freq_masks, # CLODE: +pad_mask passthrough

        seq_len, channel = eeg_signal.shape # (CW) - multiple samples packed into single batch
        batch=1

        t_shape = (
            (batch, (seq_len // self.diffusion_forcing_num_frames)+1, 1)
            if self.diffusion_forcing
            else (batch, 1, 1)
        )
        if self.diffusion_noise_schedule == "linear":
            t = torch.rand(*t_shape, device=eeg_signal.device)
        elif self.diffusion_noise_schedule == "beta":
            t = beta_sched(t_shape, device=eeg_signal.device, dtype=eeg_signal.dtype)
        elif self.diffusion_noise_schedule == "logit":
            t = logit_normal_sched(t_shape, device=eeg_signal.device, dtype=eeg_signal.dtype, 
                                    m=self.logit_normal_mean, 
                                    s=self.logit_normal_std)

        # if diffusion forcing, duplicate dim 1 to match decoder_stft seq_len such that t1 t2 t3 -> t1 t1 ... t2 t2 ... t3 t3 ..
        if self.diffusion_forcing:
            t = torch.repeat_interleave(t, self.diffusion_forcing_num_frames, dim=1)[:, :seq_len, :]

        sigma = self.global_sigma

        # Apply token dropout here to eeg_signal
        eeg_signal_masked = eeg_signal.clone()
        eeg_signal_masked[token_dropout.squeeze(-1),:] = 0.0

        # Make random noise signal. But, maintain x,y,z channel positions if you concated them in.
        noise = torch.randn_like(eeg_signal) * sigma
        if self.dont_noise_chan_xyz:
            if self.cat_chan_xyz_and_eeg:
                noise[:,:3] = eeg_signal[:,:3] # dont add noise to {x,y,z}-position channels.   
                eeg_signal_masked[:,:3] = eeg_signal[:,:3] # dont mask {x,y,z}-position channels.
            else:
                print("NOTE: EEG channel {x,y,z}-position was never concatenated into signal.")

        if self.masked_in_decoder:
            decoder_input = (1 - t) * eeg_signal_masked + t * noise # dropped out noised signals sent into decoder input.
        else:
            decoder_input = (1 - t) * eeg_signal + t * noise # non dropped outnoised signals sent into decoder input.

        decoder_targets = noise - eeg_signal


        # Print out mean and std of noise and signals and combinations of them. (Check sigma) Data_sig = 0.2 & Noise_sig = 1.0
        print_sample_noising_process = False
        if print_sample_noising_process:

            # Loop over 10 values of t between 0 and 1
            print("\n" + "="*80)
            print("Statistics for dropout_chan vs ~dropout_chan subsets")
            print("="*80)

            dropout_mask = token_dropout.squeeze(-1)

            decoder_targets_test = noise - eeg_signal

            # These should be same for all t values. So only print once.
            # eeg_signal stats
            sig_do = eeg_signal[dropout_mask]
            sig_nodo = eeg_signal[~dropout_mask]
            print(f"\n  eeg_signal [dropout]:     mean={sig_do.mean():.6f}, std={sig_do.std():.6f}")#, min={sig_do.min():.6f}, max={sig_do.max():.6f}")
            print(f"  eeg_signal [~dropout]:    mean={sig_nodo.mean():.6f}, std={sig_nodo.std():.6f}")#, min={sig_nodo.min():.6f}, max={sig_nodo.max():.6f}")

            # eeg_signal_masked stats
            sig_do = eeg_signal_masked[dropout_mask]
            sig_nodo = eeg_signal_masked[~dropout_mask]
            print(f"\n  eeg_signal_masked [dropout]:  mean={sig_do.mean():.6f}, std={sig_do.std():.6f}")#, min={sig_do.min():.6f}, max={sig_do.max():.6f}")
            print(f"  eeg_signal_masked [~dropout]: mean={sig_nodo.mean():.6f}, std={sig_nodo.std():.6f}")#, min={sig_nodo.min():.6f}, max={sig_nodo.max():.6f}")

            # noise stats
            sig_do = noise[dropout_mask]
            sig_nodo = noise[~dropout_mask]
            print(f"\n  noise [dropout]:          mean={sig_do.mean():.6f}, std={sig_do.std():.6f}")#, min={sig_do.min():.6f}, max={sig_do.max():.6f}")
            print(f"  noise [~dropout]:         mean={sig_nodo.mean():.6f}, std={sig_nodo.std():.6f}")#, min={sig_nodo.min():.6f}, max={sig_nodo.max():.6f}")

            # decoder_targets_test stats
            sig_do = decoder_targets_test[dropout_mask]
            sig_nodo = decoder_targets_test[~dropout_mask]
            print(f"\n  decoder_targets [dropout]:  mean={sig_do.mean():.6f}, std={sig_do.std():.6f}")#, min={sig_do.min():.6f}, max={sig_do.max():.6f}")
            print(f"  decoder_targets [~dropout]: mean={sig_nodo.mean():.6f}, std={sig_nodo.std():.6f}")#, min={sig_nodo.min():.6f}, max={sig_nodo.max():.6f}")

            for i, t_val in enumerate(torch.linspace(0, 1, 10)):
                # Compute noisy signal for this t value
                t_test = t_val * torch.ones_like(t)
                noisy_test = ((1 - t_test) * eeg_signal_masked + t_test * noise).squeeze(0)
                
                print(f"\n--- t = {t_val:.3f} sigma = {sigma} ---")

                # noisy_test stats
                sig_do = noisy_test[dropout_mask]
                sig_nodo = noisy_test[~dropout_mask]
                print(f"  noisy_eeg_signal_masked [dropout]:  mean={sig_do.mean():.6f}, std={sig_do.std():.6f}")#, min={sig_do.min():.6f}, max={sig_do.max():.6f}")
                print(f"  noisy_eeg_signal_masked [~dropout]: mean={sig_nodo.mean():.6f}, std={sig_nodo.std():.6f}")#, min={sig_nodo.min():.6f}, max={sig_nodo.max():.6f}")


            print("\n" + "="*80 + "\n")


            print(f"INside EEGProcessor.process, plotting sample of noisy and clean signals.")

            print(
                eeg_signal, 
                chan_pos, 
                chan_pos_discrete, 
                chan_id, 
                t_coarse, 
                seq_lens, 
                token_dropout,
                sigma,
                eeg_signal_masked,
                noise,
                decoder_input,
                decoder_targets,
                t,
            )



        out_dict = {
            "encoder_input": eeg_signal_masked, # dropout signals into encoder input.
            "decoder_input": decoder_input,     # send noised version of signal or masked signal to decoder input.
            "target": decoder_targets,
            "t": t,
            "eeg_signal": eeg_signal,                   # just passing eeg_signal through.
            "chan_pos": chan_pos,                       # just passing chan_pos through.
            "chan_pos_discrete": chan_pos_discrete,     # just passing chan_pos_discrete through.
            "chan_id": chan_id,                         # just passing chan_id through.
            "seq_lens": seq_lens,                       # just passing seq_lens through.
            "max_tc": max_tc,                           # just passing max_tc through.
            "t_coarse": t_coarse,                       # just passing t_coarse through.
            "pad_mask": pad_mask,                       # CLODE: [N,1] 1=real 0=pad, rides through to the model
        }

        return out_dict



def worker_init_fn(worker_id, seed=42, rank=0):
    """Initialize worker with unique seed."""
    # Create unique seed for this worker and rank
    worker_seed = int(seed + (1e3 * rank) + (1e6 * worker_id))

    # Set all random seeds for this worker
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

    # Set the dataset's random state
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:  # In multiprocessing
        worker_info.dataset.state = np.random.RandomState(worker_seed)


def create_pack_chans_collate_fn(target_packed_seqlen=1): #batch, 
    """
    Do Sequence packing here and in EEGDataset_v2
    """
    def pack_chans_collate_fn(batch):
        packed_batch_dict = {
            'eeg_signal':               torch.vstack([item['eeg_signal'] for item in batch[0]]),
            'chan_pos':                 torch.vstack([item['chan_pos'] for item in batch[0]]),
            'chan_pos_discrete':        torch.vstack([item['chan_pos_discrete'] for item in batch[0]]),
            'chan_id':                  torch.vstack([item['chan_id'] for item in batch[0]]),
            't_coarse':                 torch.vstack([item['t_coarse'] for item in batch[0]]),
            'token_dropout':             torch.vstack([item['token_dropout'] for item in batch[0]]),
            'pad_mask':                  torch.vstack([item['pad_mask'] for item in batch[0]]),  # CLODE [total,1]
            #
            'max_tc':                   torch.tensor([item['max_tc'] for item in batch[0]]),
            'seq_lens':                 torch.tensor([item['seq_lens'] for item in batch[0]]),
            'ids':                      torch.tensor([item['ids'] for item in batch[0]]),                
            'dataset_id':               torch.tensor([item['dataset_id'] for item in batch[0]]),  
        }
        return packed_batch_dict

    return pack_chans_collate_fn


def create_dataloader_v2(args: BCIDatasetArgs, seed, rank, timeout=200):
    if args.use_v3:  #jm
        dataset = EEGDataset_v3(args) # IterableDataset pulling from v7 mmap format!
    elif args.use_b2:
        dataset = EEGDataset_b2(args) # IterableDataset pulling from B2!
    else:
        dataset = EEGDataset_v2(args) # IterableDataset pulling from local filesystem!
        

    is_distributed = dist.is_available() and dist.is_initialized()
    sampler = None
    shuffle = args.shuffle  # Keep original shuffle intent if not distributed

    if is_distributed:
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()  # Use global rank for sampler
        print(f"Rank {global_rank}/{world_size}: Using DistributedSampler.")

    import functools
    init_fn = functools.partial(worker_init_fn, seed=seed, rank=rank)

    if args.num_workers==0:
        timeout=0 # (CW) - to pass an assertion error when debugging.


    # create sequence packing collator function
    pack_chans_collate_fn = create_pack_chans_collate_fn(args.target_packed_seqlen)


    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        worker_init_fn=init_fn,
        drop_last=is_distributed,
        timeout=timeout,
        in_order=False,
        collate_fn=pack_chans_collate_fn
    )

