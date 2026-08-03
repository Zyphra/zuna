import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
# import zarr
import numpy as np
import mne
import math
import json
from dataclasses import dataclass, field
from typing import Union, List, Optional, Any
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import random
import os
from dotenv import load_dotenv
import time
from pathlib import Path

import matplotlib.pyplot as plt

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

    # fixed-eval harness (see eval_harness.py)
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

    pad_packed_seqlen: bool = False  # if True, pad each packed seq with one all-zero document up to EXACTLY
                                     # target_packed_seqlen (fixed shapes -> no torch.compile recompiles / frag).
                                     # Requires target_packed_seqlen % encoder_latent_downsample_factor == 0.

    diffusion_forcing: bool = False
    diffusion_forcing_num_frames: int = 1

    patching_type: str = "frames"
    stft_global_sigma: Union[str, float] = 1.0
    masked_in_decoder: bool = True # If true, mask out channels in decoder input when channel is dropped. (true works, false does not)

    num_bins_discretize_xyz_chan_pos: int = 100 # Number of bins to discretize channel positions to use in 4d-RoPE. # 40 with "old" xyz_extremes, 100 with "thirteens" xyz_extremes
    chan_pos_xyz_extremes_type: str = "thirteens" # "old" for v4 dataset or "thirteens" for v5 dataset

    # v3 mmap fields — ignored by EEGDataset_v2/b2, used only when use_v3=True
    use_v3: bool = False
    filter_version: List[str] = field(default_factory=lambda: ["v3_bandpass"]) # WAS str = "v3_bandpass"                   # (use_v3: reads mmap from data_dir)
    min_quality_any: float = 0.1
    min_quality_mean: float = 0.3
    dataset_id: int = 7
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

    # v4 fields — used only by EEGDataset_v4 (.fif inference loader). Defaults are inference-friendly. #jm v4
    use_v4: bool = False                                                                              #jm v4
    v4_highpass_hz: Optional[float] = None    # None = skip highpass filter                            #jm v4
    v4_lowpass_hz: Optional[float] = None     # None = skip lowpass filter                             #jm v4
    v4_notch_hz: Optional[List[float]] = None  # None = skip notch; pass [60] for single, [50,60,100] for multi  #jm v4
    v4_montage: Optional[str] = None          # MNE montage name; used only if file has no positions   #jm v4
    v4_segment_sec: float = 10.0              # length of each segment in seconds                      #jm v4
    v4_flat_thresh: Optional[float] = None    # per-(ch,coarse-time) std threshold for flat detection  #jm v4
    v4_noise_thresh: Optional[float] = None   # per-(ch,coarse-time) std MAD multiplier for noisy det. #jm v4
    v4_require_positions: bool = True         # drop channels lacking 3D coords                        #jm v4
    v4_drop_channels: Optional[List[str]] = None  # channel names to repair (mask->interpolate) even if not flagged bad in the .fif  #jm v4
    v4_recon_save_fif: bool = True            # save model-reconstructed .fif files into dump_dir       #jm v4
    v4_recon_fill_only_masked: bool = True    # fill only masked cells (True) or whole signal (False)   #jm v4
    v4_recon_unmasked_from_original: bool = False  # unmasked cells = raw (resampled, unfiltered) volts #jm v4
    v4_recon_out_dir: Optional[str] = None  # base dir for .fif save-out (full_reconstruction/ + hybrid/ subfolders); falls back to dump_dir #jm v4
    v4_recon_seam_correct: bool = True  # re-anchor hybrid infills to neighbouring original (remove boundary jumps; DC/near-DC only) #jm v4
    v4_recon_annotate_infill: bool = True  # mark reconstructed cells on the output .fif with 'ZUNA1.1_infilled' annotations (per-channel); fully-infilled channels get a full-duration annotation and drop from info['bads'] #jm v4
    v4_recon_save_preprocessed: bool = False  # also write the post-preprocessing raw (resampled+filtered+montage+bads+upsampled channels, exactly what the model ingests) to <v4_recon_out_dir>/fif_input_preprocessed/ #jm v4
    v4_mask_dir: Optional[str] = None  # dir of per-file <base>_mask.npz (channel x sample bool, 0-based) UNIONed into the reconstruction mask (UI / manual segments) #jm v4
    v4_use_fif_annotations: bool = True  # import BAD_ time-segment annotations from the .fif into the reconstruction mask (whole-channel info['bads'] are always used) #jm v4
    v4_bad_token_overlap: float = 0.0  # a coarse token (num_fine_time_pts samples) is marked bad when the fraction of it overlapped by a BAD_ segment / manual bad_segment exceeds this; 0.0 = any overlap (widen out to whole tokens), 0.5 = only if the majority of the token is bad (rounds tight edges inward) #jm v4
    v4_filter_method: str = "fir"             # MNE filter method: "fir" (default, accurate) or "iir"   #jm v4
    # Channel upsampling: add zero-filled channels at target-montage positions and let the model        #jm v4
    # interpolate them (they are masked, like bads). int = greedy upsample to N total channels;          #jm v4
    # list[str] = add these named channels; None = disabled. Mirrors the zuna release mechanism.         #jm v4
    # Typed Any (not Union[int, List[str]]) because OmegaConf can't represent unions of containers.       #jm v4
    v4_target_channel_count: Optional[Any] = None                                                         #jm v4
    v4_upsample_montage: str = "standard_1005"  # MNE montage used as the source of target ch positions   #jm v4
    # Profiling: when True, EEGDataset_v4 accumulates per-step timing (read/filter/resample/z-score/...). #jm v4
    # Off by default so the hot per-segment loop pays nothing in production.                              #jm v4
    v4_profile: bool = False                                                                              #jm v4



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
            "random-uniform-dropout",
            "full-channel-random-dropout-train", 
            "correlated-channel-time-dropout",
            "full-time-pt-random-dropout", 
            "random-montage-channel-dropout", 
            "brain-region-channel-dropout",
            "consumer-eeg-channel-dropout"],
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
        # a. self.token_dropout_prob determines whether we do channel dropout for this sample.
        # If we do channel dropout, 
        # b. with p=0.8, we drop between 1 and N/2 chans with uniform probability.
        # c. with p=0.2, we drop between N/2 and N-1 chans with uniform probability.
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
        # a. self.token_dropout_prob determines whether we do time-point dropout for this sample.
        # If we do time-point dropout, 
        # b. Draw tc_width from a triangle distribution defined by low, mode, high.  High is constrained to be no more than 80% of sample
        # c. Draw tc_begin randomly between 0 and out a section of tc_width width centered at a random tc index.

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
        # a. self.token_dropout_prob determines whether we do time-point dropout for this sample.
        # If we do correlated channel + time-point dropout, 
        # b. Draw tc_width from a triangle distribution defined by low, mode, high.  High is constrained to be no more than 80% of sample
        # c. Draw tc_begin randomly between 0 and out a section of tc_width width centered at a random tc index.
        # d. with p=0.8, we drop between 1 and N/2 chans with uniform probability.
        # e. with p=0.2, we drop between N/2 and N-1 chans with uniform probability.

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


    elif dropout_scheme == "eval-1" or dropout_scheme == "full-channel-random-dropout-eval":
        ## NOTE: THIS FIXED DROPOUT RATE SCHEME USED FOR EVALS. FIRST, RANDOMLY DROP p*N CHANNELS.
        # CAN ALSO DROP OUT CHANNELS IN AN ORGANIZED WAY FROM THE GRID.
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




class EEGDataset_v3(IterableDataset): # loads from v7 mmap format (JSON sidecar + .dat files)
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
        self.mmap_dir               = Path(args.data_dir)  # reuses data_dir as mmap root
        self.filter_version         = args.filter_version
        self.min_quality_any        = args.min_quality_any
        self.min_quality_mean       = args.min_quality_mean
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
        self.pad_packed_seqlen      = args.pad_packed_seqlen
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


        # Duration window sampling config for EEGDataset_v3.
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
        if args.chan_pos_xyz_extremes_type == "fifteens":
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


        # Build file index from v7 metadata JSONs
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
                        "samples_per_seg":   int(round(float(m.get("quality_segment_sec", 1.0)) * int(m["fs"]))),
                    })
            except Exception as e:
                print(f"Warning: skipping {json_path.name}: {e}")
                continue

        # Flag to skip and not use pre-epoched data before it was implemented in the preprocessing pipeline.
        if self.skip_preepoched_data:
            self.file_index = [r for r in self.file_index if r["is_epoched"]==False]


        # Duration-weighted sampling probabilities (longer files sampled more often)
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

        # Soft sharding: each worker uses an independent numpy RNG seeded by rank+worker+epoch.
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
            file_idx = -1  # ensure defined even if exception fires before assignment
            try:
                # 1. Sample a file proportional to its duration
                file_idx = int(rng.choice(len(self.file_index), p=self.file_weights))
                rec = self.file_index[file_idx]
                tf  = self.num_fine_time_pts

                # 2. Sample a window duration from V3_DURATION_RANGES
                range_weights = np.array([w for _, _, w in self.V3_DURATION_RANGES])
                range_weights = range_weights / range_weights.sum()
                chosen_range  = int(rng.choice(len(self.V3_DURATION_RANGES), p=range_weights))
                lo, hi, _     = self.V3_DURATION_RANGES[chosen_range]
                win_sec       = float(rng.uniform(lo, hi))
                win_samples   = max(tf, int(round(win_sec * rec["fs"] / tf)) * tf)
                win_samples   = min(win_samples, rec["n_samples"])
                if win_samples < tf:
                    continue


                # 3. Load window from mmap (continuous or epoched)
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

                    # Debugging: Print the extracted sample bounds and shape.
                    assert bound_start <= start <= start + win_samples <= bound_stop, f"Invalid sample bounds: {bound_start=} {start=} {start + win_samples=} {bound_stop=}"
                    assert data_np.shape[1] == win_samples, f"Invalid sample shape: {data_np.shape[1]=} {win_samples=}"

                # print(f"In EEGDataset_v3.__iter__, just before Quality-based channel filter...")

                # 4. Quality-based channel filter — window-specific, two thresholds
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





                # 5. Convert to torch; build channel positions (same as EEGDataset_v2)

                eeg_t              = torch.from_numpy(data_np).float()
                chan_pos           = torch.tensor(xyz_good, dtype=torch.float32)
                chan_pos_discrete  = discretize_chan_pos(chan_pos, self.xyz_extremes, self.num_bins)
                

                # 6. Channel dropout (mirrors EEGDataset_v2 dropout schemes exactly)
                token_dropout = perform_token_dropout(dropout_scheme=self.dropout_scheme, 
                                                      token_dropout_prob=self.token_dropout_prob, 
                                                      num_fine_time_pts=self.num_fine_time_pts, 
                                                      mmap=[eeg_t],
                                                      channel_names=channel_names,
                                                      chan_pos=chan_pos)

                assert len(token_dropout)==1 



                # 7. Reshape signals (same call signature as EEGDataset_v2)
                reshaped = chop_and_reshape_signals(eeg_t, chan_pos, chan_pos_discrete, tf, self.use_coarse_time)

                if self.cat_chan_xyz_and_eeg:
                    eeg_cat = torch.cat((reshaped[1], reshaped[0]), dim=1)
                else:
                    eeg_cat = reshaped[0]

                # 8. Pack into packed_batch — yield when target_packed_seqlen is reached (mirrors EEGDataset_v2)
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
                         "pad_mask":           torch.ones(reshaped[5], 1, dtype=torch.float32),  # 1=real
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
                            "pad_mask":           torch.ones(tokens_left, 1, dtype=torch.float32),  # 1=real
                            "ids":                file_idx,
                            "dataset_id":         self.dataset_id}
                        )

                    # pad up to EXACTLY target_packed_seqlen as ONE extra all-zero
                    # document. Becomes its own block in the doc mask (isolated in
                    # attention) and is zeroed out of every loss via pad_mask.
                    cur_total = sum(item["seq_lens"] for item in packed_batch)
                    n_pad = self.target_packed_seqlen - cur_total
                    if self.pad_packed_seqlen and n_pad > 0:        # gated by config flag
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


class EEGDataset_v4(IterableDataset):  # sequential .fif inference loader
    """
    Inference-time dataset that loads .fif (or any MNE-readable) files directly.

    Differences vs EEGDataset_v3:
      - Source: .fif via MNE. No precomputed mmap / quality matrix required.
      - Sequential, deterministic windowing (no random sampling). Walks each file
        end-to-end in fixed segment_sec windows; final window may be shorter.
      - Optional inline preprocessing: highpass / lowpass / notch / resample to
        self.sample_rate. All three filter knobs default to None (= skip), so the
        loader trusts the file unless told otherwise.
      - Bad-mask is computed at load time from MNE info['bads'] + BAD_* annotations,
        optionally augmented with heuristic flat / high-σ detection. Bad cells are
        fed to the model as dropout tokens — they are *masked*, not removed, so
        the model interpolates them.
      - Each yielded segment dict carries reconstruction metadata (seg_mean,
        seg_std, good_ch, fif_path, seg_start, seg_end) so downstream code can
        invert the per-segment z-score and reassemble the file.

    Output dict format matches EEGDataset_v3 exactly (same keys), plus the extra
    reconstruction-metadata keys above, so model code does not need to branch.
    """

    def __init__(self, args: BCIDatasetArgs):
        print(f"Inside EEGDataset_v4 with {args.data_dir=}")
        self.data_dir              = Path(args.data_dir)
        self.glob_filter           = args.glob_filter
        self.dataset_id            = args.dataset_id
        self.seed                  = args.seed
        self.num_workers           = args.num_workers
        self._current_epoch        = 0
        self.num_fine_time_pts     = args.num_fine_time_pts
        self.sample_rate           = args.sample_rate
        self.use_coarse_time       = args.use_coarse_time
        self.cat_chan_xyz_and_eeg  = args.cat_chan_xyz_and_eeg
        self.target_packed_seqlen  = args.target_packed_seqlen
        self.token_dropout_prob    = args.token_dropout_prob
        self.dropout_scheme        = args.dropout_scheme
        self.num_bins              = args.num_bins_discretize_xyz_chan_pos
        self.do_avg_ref            = args.do_avg_ref
        self.z_score_type          = args.z_score_type

        # v4-specific
        self.highpass_hz           = args.v4_highpass_hz
        self.lowpass_hz            = args.v4_lowpass_hz
        self.notch_hz              = args.v4_notch_hz
        self.montage               = args.v4_montage
        self.segment_sec           = args.v4_segment_sec
        self.flat_thresh           = args.v4_flat_thresh
        self.noise_thresh          = args.v4_noise_thresh
        self.require_positions     = args.v4_require_positions
        self.drop_channels         = getattr(args, "v4_drop_channels", None)  #jm v4
        self.mask_dir              = getattr(args, "v4_mask_dir", None)  #jm v4 (per-file external bad-mask dir)
        self._external_mask        = None  #jm v4 (per-file, set in _prepare_raw)
        self.use_fif_annotations   = getattr(args, "v4_use_fif_annotations", True)  #jm v4
        self.bad_token_overlap     = float(getattr(args, "v4_bad_token_overlap", 0.0))  #jm v4
        self.recon_fill_only_masked = args.v4_recon_fill_only_masked  #jm v4
        self.recon_unmasked_from_original = getattr(args, "v4_recon_unmasked_from_original", False)  #jm v4
        self.recon_out_dir         = getattr(args, "v4_recon_out_dir", None)  #jm v4
        self.recon_save_preprocessed = getattr(args, "v4_recon_save_preprocessed", False)  #jm v4
        self.filter_method         = getattr(args, "v4_filter_method", "fir")  #jm v4
        self.target_channel_count  = getattr(args, "v4_target_channel_count", None)  #jm v4
        self.upsample_montage      = getattr(args, "v4_upsample_montage", "standard_1005")  #jm v4
        self.profile_enabled       = getattr(args, "v4_profile", False)  #jm v4
        # raw_info_registry: keeps mne.Info per source file so the FifReconstructor    #jm v4
        # can rebuild .fif outputs with the original metadata (channel names,           #jm v4
        # sfreq, montage, etc). num_workers must be 0 for this to be visible to the     #jm v4
        # main process; for num_workers>0 the registry would live in each worker only.  #jm v4
        self.raw_info_registry: dict = {}                       #jm v4

        # Timing accumulators. _prepare_raw_time = total seconds spent loading+
        # filtering+resampling .fif files across the run. _segment_build_time =
        # total seconds spent per-segment (z-score, bad-mask, chop). Read by the
        # eeg_eval main loop to print summary lines.
        self._prepare_raw_time = 0.0
        self._segment_build_time = 0.0
        self._n_segments_yielded = 0
        self._n_files_loaded = 0

        # Per-step timing (only populated when v4_profile=True). Keys cover both the
        # per-file _prepare_raw sub-steps and the per-segment __iter__ sub-steps.
        # Surfaced to the main process via the batch dict (see __iter__), since with
        # num_workers>0 these live in worker processes only.
        self._step_times = {
            # _prepare_raw sub-steps
            "read": 0.0, "pick": 0.0, "montage": 0.0, "drop_nopos": 0.0,
            "resample": 0.0, "unfiltered_snapshot": 0.0, "filter": 0.0,
            "notch": 0.0, "upsample": 0.0,
            # per-segment sub-steps
            "slice": 0.0, "avg_ref": 0.0, "z_score": 0.0, "bad_mask": 0.0,
            "to_torch": 0.0, "token_dropout": 0.0, "chop_reshape": 0.0, "pack": 0.0,
        }
        # Tracks which channel indices were added by upsampling (zero-filled, masked).
        # Reset per file in _prepare_raw; consumed by the avg-ref step in __iter__.
        self._upsampled_ch_mask = None

        # xyz_extremes — identical to V3
        if args.chan_pos_xyz_extremes_type == "fifteens":
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

        # File discovery — supports single-file or directory roots
        if self.data_dir.is_file():
            self.file_paths = [self.data_dir]
        else:
            patterns = self.glob_filter if isinstance(self.glob_filter, (list, tuple)) else [self.glob_filter]
            seen = set()
            self.file_paths = []
            for pat in patterns:
                for p in sorted(self.data_dir.glob(pat)):
                    if p.is_file() and p not in seen:
                        seen.add(p)
                        self.file_paths.append(p)
        print(f"In EEGDataset_v4.__init__, {len(self.file_paths)} file(s) matched glob_filter={self.glob_filter} under {self.data_dir}")

    def __len__(self):
        return 10**10

    def set_epoch(self, epoch):
        self._current_epoch = epoch

    # ------------------------------------------------------------------ #
    # File preparation: load + pick EEG + montage + filter + resample
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # Profiling helper: accumulate elapsed into _step_times[key], return
    # a fresh perf_counter cursor so calls can be chained.
    # ------------------------------------------------------------------ #
    def _tick(self, key, t0):
        now = time.perf_counter()
        self._step_times[key] += now - t0
        return now

    # ------------------------------------------------------------------ #
    # Channel upsampling: append zero-filled channels at target-montage
    # positions and mark them bad, so the model interpolates them. Mirrors
    # zuna/src/zuna/preprocessing/interpolation.py (greedy + named modes).
    # Sets self._upsampled_ch_mask (bool over the FINAL channel list).
    # ------------------------------------------------------------------ #
    def _maybe_upsample_channels(self, raw):
        n_orig = len(raw.ch_names)
        self._upsampled_ch_mask = np.zeros(n_orig, dtype=bool)
        target = self.target_channel_count
        if target is None:
            return raw

        montage = mne.channels.make_standard_montage(self.upsample_montage)
        mpos = montage.get_positions()["ch_pos"]   # {name: (x,y,z)}
        present_lower = {c.lower() for c in raw.ch_names}

        # Candidate montage channels not already present, with finite positions.
        candidates = [
            (name, np.asarray(pos, dtype=np.float64))
            for name, pos in mpos.items()
            if name.lower() not in present_lower
            and np.all(np.isfinite(pos)) and not np.allclose(pos, 0.0)
        ]

        if isinstance(target, (list, tuple)):
            # Named mode: keep only requested names (case-insensitive), preserve order.
            want = {str(n).lower() for n in target}
            to_add = [(nm, p) for (nm, p) in candidates if nm.lower() in want]
        else:
            # Greedy mode: add (target - current) channels, furthest-first from the
            # existing electrode set to maximize spatial coverage (zuna algorithm).
            n_to_add = int(target) - n_orig
            if n_to_add <= 0:
                return raw
            cur_pos = np.array([ch["loc"][:3] for ch in raw.info["chs"]], dtype=np.float64)
            valid = ~np.all(cur_pos == 0, axis=1) & ~np.isnan(cur_pos).any(axis=1)
            cur_pos = cur_pos[valid] if valid.any() else cur_pos
            scored = []
            for nm, p in candidates:
                d = np.linalg.norm(cur_pos - p[None, :], axis=1).min() if len(cur_pos) else 0.0
                scored.append((d, nm, p))
            scored.sort(key=lambda s: s[0], reverse=True)   # furthest first
            to_add = [(nm, p) for (_, nm, p) in scored[:n_to_add]]

        if not to_add:
            return raw

        add_names = [nm for nm, _ in to_add]
        add_pos   = np.array([p for _, p in to_add], dtype=np.float64)
        zeros = np.zeros((len(add_names), raw.n_times), dtype=np.float64)
        add_info = mne.create_info(add_names, raw.info["sfreq"], ch_types="eeg")
        add_raw = mne.io.RawArray(zeros, add_info, verbose="ERROR")
        # Set 3D positions on the new channels' loc[:3].
        for i, ch in enumerate(add_raw.info["chs"]):
            ch["loc"][:3] = add_pos[i]
        raw.add_channels([add_raw], force_update_info=True)
        # Mark the new channels bad so they are masked everywhere -> model interpolates.
        raw.info["bads"] = list(raw.info["bads"]) + add_names

        mask = np.zeros(len(raw.ch_names), dtype=bool)
        name_to_idx = {nm: i for i, nm in enumerate(raw.ch_names)}
        for nm in add_names:
            mask[name_to_idx[nm]] = True
        self._upsampled_ch_mask = mask
        print(f"  [v4] upsampled {n_orig} -> {len(raw.ch_names)} channels "
              f"(added {len(add_names)}: {add_names[:8]}{'…' if len(add_names) > 8 else ''})")
        return raw

    def _prepare_raw(self, fif_path):
        _t_prep_start = time.perf_counter()
        _t = _t_prep_start
        prof = self.profile_enabled
        raw = mne.io.read_raw(str(fif_path), preload=True, verbose="ERROR")
        if prof: _t = self._tick("read", _t)
        raw.pick("eeg")
        if prof: _t = self._tick("pick", _t)

        # Apply named montage only as a FALLBACK when the file lacks positions.
        if self.montage is not None:
            _pre_locs = np.array([ch["loc"][:3] for ch in raw.info["chs"]])
            _has_pos  = ~np.all(_pre_locs == 0, axis=1) & ~np.isnan(_pre_locs).any(axis=1)
            if not _has_pos.all():
                raw.set_montage(self.montage, on_missing="warn", match_case=False)
            else:
                print(f"  [v4] all {len(raw.ch_names)} ch have file-provided positions; skipping montage '{self.montage}'")
        if prof: _t = self._tick("montage", _t)

        # Force-mark user-requested channels as bad (case-insensitive) so the model
        # interpolates them even if the .fif didn't flag them in info['bads']. Reuses
        # the same whole-channel bad path as file-marked bads (_compute_bad_mask_2d).
        if self.drop_channels:
            _name_by_lower = {c.lower(): c for c in raw.ch_names}
            _to_bad  = [_name_by_lower[n.lower()] for n in self.drop_channels if n.lower() in _name_by_lower]
            _missing = [n for n in self.drop_channels if n.lower() not in _name_by_lower]
            if _missing:
                print(f"  [v4] v4_drop_channels not present in {Path(fif_path).name}: {_missing}")
            raw.info["bads"] = list(dict.fromkeys(list(raw.info["bads"]) + _to_bad))
            if _to_bad:
                print(f"  [v4] repairing user-requested channels (mask->interpolate): {_to_bad}")

        # Drop channels with no 3D position
        chs_loc = np.array([ch["loc"][:3] for ch in raw.info["chs"]])
        valid_pos = ~np.all(chs_loc == 0, axis=1) & ~np.isnan(chs_loc).any(axis=1)
        if self.require_positions and not valid_pos.all():
            bad_pos_names = [raw.ch_names[i] for i, ok in enumerate(valid_pos) if not ok]
            print(f"  [v4] dropping {len(bad_pos_names)} channels without 3D coords: {bad_pos_names[:8]}{'…' if len(bad_pos_names) > 8 else ''}")
            raw.pick([raw.ch_names[i] for i, ok in enumerate(valid_pos) if ok])
            if len(raw.ch_names) < 3:
                raise RuntimeError(f"{fif_path}: <3 channels with valid 3D coords")
        if prof: _t = self._tick("drop_nopos", _t)

        # Resample FIRST. MNE's resample() includes its own anti-alias lowpass so
        # this is safe; running the user filter AFTER resample is much faster (the
        # 0.01 Hz FIR is ~4× shorter at 256 Hz than at 1000 Hz). Has no meaningful
        # effect on signal content vs the old filter-then-resample order.
        if int(round(raw.info["sfreq"])) != self.sample_rate:
            raw.resample(self.sample_rate, verbose="ERROR")
        if prof: _t = self._tick("resample", _t)

        # Snapshot the resampled-but-unfiltered data BEFORE applying highpass/notch.
        # Used by FifReconstructor when v4_recon_unmasked_from_original=True so the
        # unmasked cells of the recon come straight from the file rather than from the
        # filtered-then-inverse-zscored model input path.
        unfiltered_data = None
        if self.recon_unmasked_from_original:
            unfiltered_data = raw.get_data().astype(np.float32, copy=True)
        if prof: _t = self._tick("unfiltered_snapshot", _t)

        # Apply user filter(s).
        filter_method = self.filter_method if self.filter_method in ("fir", "iir") else "fir"
        if self.highpass_hz is not None or self.lowpass_hz is not None:
            raw.filter(l_freq=self.highpass_hz, h_freq=self.lowpass_hz,
                       method=filter_method, verbose="ERROR")
        if prof: _t = self._tick("filter", _t)
        if self.notch_hz is not None:
            raw.notch_filter(self.notch_hz, method=filter_method, verbose="ERROR")
        if prof: _t = self._tick("notch", _t)

        # Channel upsampling (adds zero-filled, masked channels at montage positions).
        # If recon_unmasked_from_original snapshot was taken above, extend it with
        # zero rows for the added channels so shapes stay aligned downstream.
        n_before_upsample = len(raw.ch_names)
        raw = self._maybe_upsample_channels(raw)
        if unfiltered_data is not None and len(raw.ch_names) > n_before_upsample:
            pad = np.zeros((len(raw.ch_names) - n_before_upsample, unfiltered_data.shape[1]),
                           dtype=unfiltered_data.dtype)
            unfiltered_data = np.concatenate([unfiltered_data, pad], axis=0)
        if prof: _t = self._tick("upsample", _t)

        # Register the (post-preprocessing) info so the FifReconstructor can use it.
        self.raw_info_registry[str(fif_path)] = raw.info.copy()

        # Optionally persist the fully-preprocessed continuous raw (exactly what the model
        # ingests: resampled + filtered + montage + info['bads'] + upsampled channels) so it
        # can be compared against the raw input and the reconstruction outputs.
        if self.recon_save_preprocessed and self.recon_out_dir is not None:
            pre_dir = Path(self.recon_out_dir) / "fif_input_preprocessed"
            pre_dir.mkdir(parents=True, exist_ok=True)
            base = Path(fif_path).stem.replace("_raw", "")
            raw.save(str(pre_dir / f"{base}_raw.fif"), overwrite=True, verbose="ERROR")

        # Optional per-file external bad-mask (UI / manual bad_segments), aligned to this raw.  #jm v4
        self._external_mask = self._load_external_mask(fif_path, raw) if self.mask_dir else None

        elapsed = time.perf_counter() - _t_prep_start                                  #jm v4
        self._prepare_raw_time += elapsed                                              #jm v4
        self._n_files_loaded += 1                                                      #jm v4
        print(f"[v4 timing] _prepare_raw {Path(fif_path).name}: {elapsed*1000:.1f} ms")  #jm v4

        return raw, unfiltered_data

    def _load_external_mask(self, fif_path, raw):
        """Load <mask_dir>/<base>_mask.npz (bool mask + ch_names + sfreq [+ num_fine_time_pts]) and
        align it to the current raw: rows matched by channel name, time expanded to raw.n_times. Masks
        are stored at TOKEN resolution (one column per num_fine_time_pts samples) — detected via the
        stored num_fine_time_pts and expanded back to per-sample here; per-sample or other-duration
        masks are still accepted (nearest-resample). Treated as 0-based / data-relative. Returns
        (C, n_times) bool aligned to raw.ch_names, or None if the file has no mask.  #jm v4"""
        base = Path(fif_path).stem.replace("_raw", "")
        mpath = Path(self.mask_dir) / f"{base}_mask.npz"
        if not mpath.exists():
            print(f"  [v4] no external mask for {base} in {self.mask_dir}")
            return None
        z = np.load(str(mpath), allow_pickle=True)
        em = np.asarray(z["mask"]).astype(bool)                       # (C_ext, W)
        enames = [str(x) for x in z["ch_names"]]
        N = raw.n_times
        W = em.shape[1]
        tf = int(z["num_fine_time_pts"]) if "num_fine_time_pts" in z.files else None
        # Map each output sample -> a source column (token-, sample-, or other-duration-resolution).
        if tf is not None and W == (N + tf - 1) // tf and W != N:
            idx = np.minimum(np.arange(N) // tf, W - 1); res = "token"
        elif W == N:
            idx = np.arange(N); res = "sample"
        else:
            idx = np.minimum((np.arange(N) * (W / N)).astype(int), W - 1); res = "resampled"
        row = {n: i for i, n in enumerate(enames)}
        aligned = np.zeros((len(raw.ch_names), N), dtype=bool)
        for i, ch in enumerate(raw.ch_names):
            if ch in row:
                aligned[i] = em[row[ch]][idx]
        print(f"  [v4] external mask {mpath.name} [{res}]: {100 * aligned.mean():.1f}% cells "
              f"(aligned to {len(raw.ch_names)} ch)")
        return aligned

    # ------------------------------------------------------------------ #
    # 2D bad-mask: shape (n_channels, n_coarse_time_bins) for the segment
    # ------------------------------------------------------------------ #
    def _overlap_tokens(self, ovl_start, ovl_end, tf, n_coarse):
        """Token indices in [0, n_coarse) whose overlap with the sample span
        [ovl_start, ovl_end) exceeds self.bad_token_overlap (a fraction of the tf-sample token).

        thr=0.0 -> any overlap counts (span widens out to whole tokens, the default);
        thr=0.5 -> only tokens more than half covered (tight edges round inward).
        """
        first = max(0, ovl_start // tf)
        last  = min(n_coarse - 1, (ovl_end - 1) // tf)
        if last < first:
            return np.empty(0, dtype=np.int64)
        toks  = np.arange(first, last + 1)
        tok_s = toks * tf
        overlap = np.minimum(ovl_end, tok_s + tf) - np.maximum(ovl_start, tok_s)
        return toks[overlap > self.bad_token_overlap * tf]

    def _compute_bad_mask_2d(self, data_seg, raw, seg_start_sample, n_coarse):
        """
        data_seg : (C, T_samples)  -- ALREADY z-scored for the segment, so std is ~1 by default.
        raw      : the prepared MNE Raw (for info['bads'] and annotations).
        seg_start_sample : sample index of segment start in the (resampled) Raw.
        n_coarse : number of tf-sized coarse-time bins in the segment.

        Returns bad_2d : (C, n_coarse) bool — True where (channel, coarse-time-bin) is bad.
        """
        C, T = data_seg.shape
        tf = self.num_fine_time_pts
        bad_2d = np.zeros((C, n_coarse), dtype=bool)

        # 1. Whole-channel bads (MNE info['bads'])
        ch_names = raw.ch_names
        for bad_name in raw.info.get("bads", []):
            if bad_name in ch_names:
                bad_2d[ch_names.index(bad_name), :] = True

        # 2. Bad time annotations from the .fif (BAD_* descriptions). Toggle with
        #    v4_use_fif_annotations. Onset->sample mapping accounts for first_samp (absolute frame
        #    when orig_time is set, or when onset >= duration; else already 0-based).  #jm v4
        if self.use_fif_annotations:
            sfreq = raw.info["sfreq"]
            first_samp = raw.first_samp
            has_orig_time = raw.annotations.orig_time is not None
            dur_s = raw.n_times / sfreq
            for ann in raw.annotations:
                if not str(ann["description"]).upper().startswith("BAD"):
                    continue
                onset = ann["onset"]
                absolute = has_orig_time or (onset >= dur_s)   # onset beyond duration can't be 0-based
                off = first_samp if absolute else 0
                ann_start = int(round(onset * sfreq)) - off
                ann_end   = int(round((onset + ann["duration"]) * sfreq)) - off
                # Intersect with this segment
                ovl_start = max(0, ann_start - seg_start_sample)
                ovl_end   = min(T, ann_end   - seg_start_sample)
                if ovl_end > ovl_start:
                    # Tokens covered by this annotation, honouring the overlap threshold.
                    toks = self._overlap_tokens(ovl_start, ovl_end, tf, n_coarse)
                    if toks.size:
                        # Channel-specific bad segments: MNE stores a per-annotation ch_names
                        # tuple (MNE >= 1.0). An empty tuple means the segment applies to ALL
                        # channels (a global bad span); a populated tuple restricts it to those
                        # channels only. Names not present in this file are ignored.
                        ann_chs = ann["ch_names"] if "ch_names" in ann else ()
                        if ann_chs:
                            rows = [ch_names.index(cn) for cn in ann_chs if cn in ch_names]
                            if rows:
                                bad_2d[np.ix_(rows, toks)] = True
                        else:
                            bad_2d[:, toks] = True

        # 2b. External per-file mask (UI / manual bad_segments): 0-based, UNION into bad_2d.  #jm v4
        ext = self._external_mask
        if ext is not None and ext.shape[0] == C:
            usable = n_coarse * tf
            seg = ext[:, seg_start_sample:seg_start_sample + usable]   # (C, <=usable)
            if seg.shape[1] == usable:
                # Same overlap rule as fif annotations: a token is bad when its bad-sample
                # fraction exceeds bad_token_overlap (0.0 -> any bad sample, the default).
                frac = seg.reshape(C, n_coarse, tf).mean(axis=2)
                bad_2d |= frac > self.bad_token_overlap

        # 3. Heuristic flat / noisy detection on the z-scored segment (per-bin std)
        if self.flat_thresh is not None or self.noise_thresh is not None:
            usable = (n_coarse * tf)
            x = data_seg[:, :usable].reshape(C, n_coarse, tf)
            bin_std = x.std(axis=2)                       # (C, n_coarse)
            if self.flat_thresh is not None:
                bad_2d |= (bin_std < self.flat_thresh)
            if self.noise_thresh is not None:
                bad_2d |= (bin_std > self.noise_thresh)

        return bad_2d

    # ------------------------------------------------------------------ #
    # Main iteration: walk every file sequentially in fixed-size windows
    # ------------------------------------------------------------------ #
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers_per_rank = worker_info.num_workers if worker_info else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        global_worker_id = rank * num_workers_per_rank + worker_id

        # Worker seed (used only for perform_token_dropout's RNG)
        if self.seed is not None:
            worker_seed = int(self.seed + (1e3 * rank) + (1e6 * worker_id) + (1e9 * self._current_epoch))
        else:
            worker_seed = int(time.time() * 1000) % (2**31) + global_worker_id
        rng = np.random.default_rng(worker_seed)
        random.seed(worker_seed)

        # Soft-shard files across (rank * num_workers_per_rank) workers
        n_global_workers = max(1, world_size * num_workers_per_rank)
        my_files = [p for i, p in enumerate(self.file_paths) if i % n_global_workers == global_worker_id]

        # Packing state — persists ACROSS files so we maximally fill batches
        seqlen_accum = 0
        packed_batch = []
        tf = self.num_fine_time_pts
        prof = self.profile_enabled

        for file_idx_global, fif_path in enumerate(my_files):
            try:
                raw, unfiltered_full = self._prepare_raw(fif_path)
            except Exception as e:
                import traceback
                print(f"Error preparing {fif_path}: {e}\n{traceback.format_exc()}")
                continue

            data_full = raw.get_data().astype(np.float32, copy=False)   # (C, T) in volts
            n_channels, n_total = data_full.shape
            positions = np.array([ch["loc"][:3] for ch in raw.info["chs"]], dtype=np.float32)  # (C, 3)
            channel_names_all = list(raw.ch_names)

            # File-level stats (for downstream reference; reconstruction itself
            # only needs the per-segment μ/σ since we don't z-score at file level).
            file_mean = data_full.mean(axis=1).astype(np.float32)
            file_std  = data_full.std(axis=1).astype(np.float32) + 1e-6

            # Segment length in samples, snapped to a multiple of tf
            target_seg_samples = max(tf, int(round(self.segment_sec * self.sample_rate) // tf) * tf)

            seg_starts = list(range(0, n_total, target_seg_samples))

            for seg_start in seg_starts:
                _t_seg_start = time.perf_counter()
                _t = _t_seg_start
                seg_end = min(seg_start + target_seg_samples, n_total)
                seg_samples = seg_end - seg_start
                # Snap segment length down to a multiple of tf (allows shorter final segment)
                seg_samples = (seg_samples // tf) * tf
                if seg_samples < tf:
                    continue
                seg_end = seg_start + seg_samples

                # 1. Slice
                data_np = data_full[:, seg_start:seg_end].copy()
                if prof: _t = self._tick("slice", _t)

                # 2. Average reference (per-segment) — buffer the offset so the
                # reconstructor can add it back when inverting the normalization.
                # Upsampled channels are zero-filled; exclude them from the mean so
                # the reference isn't diluted by the added zero rows.
                if self.do_avg_ref:
                    if self._upsampled_ch_mask is not None and self._upsampled_ch_mask.any():
                        _real = ~self._upsampled_ch_mask
                        avg_ref_offset = data_np[_real].mean(axis=0)
                    else:
                        avg_ref_offset = data_np.mean(axis=0)         # (T_seg,)
                    data_np = data_np - avg_ref_offset[None, :]
                else:
                    avg_ref_offset = np.zeros(data_np.shape[1], dtype=np.float32)
                if prof: _t = self._tick("avg_ref", _t)

                # 3. Per-segment z-score (matches V3 contract — model sees ~unit-variance input)
                if self.z_score_type == "across_channel":
                    seg_mean = data_np.mean(axis=1, keepdims=True)    # (C, 1)
                    seg_std  = data_np.std(axis=1, keepdims=True) + 1e-6
                    data_np  = (data_np - seg_mean) / seg_std
                elif self.z_score_type == "across_sample":
                    g_mean = data_np.mean()
                    g_std  = data_np.std() + 1e-6
                    seg_mean = np.full((n_channels, 1), g_mean, dtype=np.float32)
                    seg_std  = np.full((n_channels, 1), g_std,  dtype=np.float32)
                    data_np  = (data_np - g_mean) / g_std
                elif self.z_score_type == "none":
                    seg_mean = np.zeros((n_channels, 1), dtype=np.float32)
                    seg_std  = np.ones((n_channels, 1),  dtype=np.float32)
                else:
                    raise ValueError(f"Invalid z_score_type: {self.z_score_type}")
                if prof: _t = self._tick("z_score", _t)

                # 4. Bad-mask: MNE bads + annotations + (optional) heuristic detection
                n_coarse = seg_samples // tf
                bad_2d = self._compute_bad_mask_2d(data_np, raw, seg_start, n_coarse)  # (C, n_coarse)
                if prof: _t = self._tick("bad_mask", _t)

                # 5. To torch
                eeg_t             = torch.from_numpy(data_np).float()
                chan_pos          = torch.from_numpy(positions).float()
                chan_pos_discrete = discretize_chan_pos(chan_pos, self.xyz_extremes, self.num_bins)
                channel_names     = channel_names_all   # unchanged this segment (no per-segment drops in v4)
                if prof: _t = self._tick("to_torch", _t)

                # 6. Random token dropout (matches V3 hook; typically off in inference via token_dropout_prob < 0)
                token_dropout = perform_token_dropout(
                    dropout_scheme=self.dropout_scheme,
                    token_dropout_prob=self.token_dropout_prob,
                    num_fine_time_pts=tf,
                    mmap=[eeg_t],
                    channel_names=channel_names,
                    chan_pos=chan_pos,
                )
                assert len(token_dropout) == 1
                if prof: _t = self._tick("token_dropout", _t)

                # 7. Chop + reshape (identical to V3)
                reshaped = chop_and_reshape_signals(eeg_t, chan_pos, chan_pos_discrete, tf, self.use_coarse_time)
                if self.cat_chan_xyz_and_eeg:
                    eeg_cat = torch.cat((reshaped[1], reshaped[0]), dim=1)
                else:
                    eeg_cat = reshaped[0]
                if prof: _t = self._tick("chop_reshape", _t)

                # 8. Build per-token dropout_bool from (a) random dropout + (b) bad-mask
                chan_id_r    = reshaped[3]
                t_coarse_r   = reshaped[4]
                dropout_bool = torch.zeros_like(chan_id_r, dtype=torch.bool)
                for cd, td in token_dropout[0]:
                    dropout_bool[(chan_id_r == cd) & (t_coarse_r == td)] = True
                # OR-in the bad-mask (channel_idx, coarse-time-bin)
                bad_2d_t = torch.from_numpy(bad_2d)                    # (C, n_coarse)
                dropout_bool |= bad_2d_t[chan_id_r.long(), t_coarse_r.long()]

                # 9. Pack into packed_batch — yield when target_packed_seqlen is hit
                seg_dict = {
                    "eeg_signal":         eeg_cat,
                    "chan_pos":           reshaped[1],
                    "chan_pos_discrete":  reshaped[2],
                    "chan_id":            chan_id_r,
                    "t_coarse":           t_coarse_r,
                    "seq_lens":           reshaped[5],
                    "max_tc":             t_coarse_r.max().item() + 1,
                    "token_dropout":      dropout_bool,
                    "pad_mask":           torch.ones(reshaped[5], 1, dtype=torch.float32),  # v4: whole segments are all-real -> satisfies zuna collate
                    "ids":                file_idx_global,
                    "dataset_id":         self.dataset_id,
                    # v4 reconstruction metadata
                    "seg_mean":           torch.from_numpy(seg_mean.squeeze(1)),         # (C,)
                    "seg_std":            torch.from_numpy(seg_std.squeeze(1)),          # (C,)
                    "avg_ref_offset":     torch.from_numpy(avg_ref_offset.astype(np.float32)),  # (T_seg,)
                    "file_mean":          torch.from_numpy(file_mean),                   # (C,)
                    "file_std":           torch.from_numpy(file_std),                    # (C,)
                    "channel_names":      channel_names,
                    "fif_path":           str(fif_path),
                    "seg_start":          int(seg_start),
                    "seg_end":            int(seg_end),
                    "sfreq":              float(self.sample_rate),
                    # mne.Info travels with every segment so FifReconstructor can
                    # work when num_workers > 0 (the dataset's raw_info_registry isn't
                    # shared across worker processes; the batch queue is).
                    "raw_info":           raw.info,
                }

                # If recon_unmasked_from_original is on, ship the resampled-but-
                # unfiltered volts for this segment so FifReconstructor can use them
                # for unmasked cells in place of the inverse-zscored model input.
                if unfiltered_full is not None:
                    seg_dict["unfiltered_volts"] = torch.from_numpy(
                        unfiltered_full[:, seg_start:seg_end].copy()
                    )

                if prof: _t = self._tick("pack", _t)
                self._segment_build_time += time.perf_counter() - _t_seg_start
                self._n_segments_yielded += 1

                # Ship a cumulative per-worker timing snapshot so the main process can
                # aggregate per-step timings even with num_workers>0 (the dataset's
                # own counters live in worker processes). Mirrors the raw_info /
                # unfiltered_volts batch-dict-shipping pattern. One dict per segment;
                # the collate fn keeps only the last (largest) snapshot per batch.
                if prof:
                    seg_dict["v4_step_times"] = {
                        "worker_id": int(global_worker_id),
                        "n_segments": int(self._n_segments_yielded),
                        "n_files": int(self._n_files_loaded),
                        "prepare_raw_time": float(self._prepare_raw_time),
                        "segment_build_time": float(self._segment_build_time),
                        "step_times": dict(self._step_times),
                    }

                if seqlen_accum + reshaped[5] <= self.target_packed_seqlen:
                    seqlen_accum += reshaped[5]
                    packed_batch.append(seg_dict)
                else:
                    # This segment would overflow — yield current batch and start a new one
                    # with this segment as its first member (do NOT split the segment, so that
                    # reconstruction boundaries stay clean).
                    if packed_batch:
                        yield packed_batch
                    seqlen_accum = reshaped[5]
                    packed_batch = [seg_dict]

        # End of all files for this worker — yield any leftover partial batch
        if packed_batch:
            yield packed_batch


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


class FifReconstructor:
    """
    Buffers per-segment model reconstructions during an inference run and writes
    them out as continuous .fif files at the end.

    Usage from eeg_eval.py:
        rec = FifReconstructor(
            output_dir=args.dump_dir,
            raw_info_registry=data_loader.dataset.raw_info_registry,
            fill_only_masked=args.data.v4_recon_fill_only_masked,
            num_fine_time_pts=args.data.num_fine_time_pts,
        )
        # ... in the eval loop, after `unwrap_all_the_signals(...)`:
        rec.add_batch(
            batch=batch,
            model_signal_input_unwrapped=model_signal_input_unwrapped,
            model_signal_output_unwrapped=model_signal_output_unwrapped,
            channel_id_unwrapped=channel_id_unwrapped,
            t_coarse_unwrapped=t_coarse_unwrapped,
        )
        # ... after the loop:
        rec.save_all()

    Notes:
      - All math runs in numpy. Model outputs are detached + .cpu().numpy() by the
        time they reach `add_batch` (via unwrap_all_the_signals).
      - save_all() writes TWO artifacts per source file:
          <base>_recon_raw.fif   — pure model output (model everywhere)
          <base>_hybrid_raw.fif  — original on kept cells, model output on dropped cells
        The infilled (channel, time) cells are recorded as ZUNA1.1_infilled annotations on
        both files (see _infill_annotations), so no separate mask .npz is written.
        The hybrid's "original" is the raw (resampled, unfiltered) file signal when
        v4_unfiltered_volts is shipped (v4_recon_unmasked_from_original=true); otherwise
        it falls back to the filtered model-input signal.
      - Segments are grouped by fif_path and sorted by seg_start; gaps between
        segments (if any) stay NaN in the signals and False in the mask.
    """
    def __init__(self, output_dir, raw_info_registry, fill_only_masked, num_fine_time_pts,
                 data_norm=1.0, unmasked_from_original=False, seam_correct=True,
                 annotate_infill=True):
        from pathlib import Path
        self.output_dir = Path(output_dir)
        # Two output subfolders (jonas): full model reconstruction, and hybrid
        # (original everywhere except the masked/repaired cells).
        self.full_dir   = self.output_dir / "full_reconstruction"
        self.hybrid_dir = self.output_dir / "hybrid"
        self.full_dir.mkdir(parents=True, exist_ok=True)
        self.hybrid_dir.mkdir(parents=True, exist_ok=True)
        self.raw_info_registry = raw_info_registry
        self.fill_only_masked = fill_only_masked
        self.tf = int(num_fine_time_pts)
        self.data_norm = float(data_norm) if data_norm else 1.0
        # When True, unmasked cells of the recon come straight from the resampled-but-
        # unfiltered file data shipped in batch['v4_unfiltered_volts']. Only masked
        # cells go through the inverse-zscore + model-output path. fill_only_masked
        # is implied to be True in this mode.
        self.unmasked_from_original = bool(unmasked_from_original)
        # Seam correction for the hybrid: the model is trained on high-pass / per-segment
        # z-scored data, so an infilled span carries no slow drift and can sit at a different
        # DC/level than the surrounding original -> a visible jump at the boundary. When on,
        # each infilled span in the hybrid is re-anchored (linear deramp) so its ends meet the
        # neighbouring original samples. Only affects DC/near-DC; the PSD bands are untouched.
        self.seam_correct = bool(seam_correct)
        # When True, save_all attaches per-channel 'ZUNA1.1_infilled' annotations to the
        # output .fif marking exactly the cells the model reconstructed (from full_mask).
        self.annotate_infill = bool(annotate_infill)
        # buffer: { fif_path : list of (seg_start, seg_end, signal_array (C_seg, T_seg), channel_names) }
        self.buffer: dict = {}

    def add_batch(self, batch, model_signal_input_unwrapped, model_signal_output_unwrapped,
                  channel_id_unwrapped, t_coarse_unwrapped):
        # Only proceed for V4 batches (which carry the per-segment metadata).
        if 'v4_fif_path' not in batch:
            return
        # Lazily register raw_info from incoming batches. Needed when num_workers>0
        # because the dataset's raw_info_registry lives in worker processes and is
        # invisible to the main process. With num_workers=0 this is a no-op since the
        # dataset already populated the same dict.
        if 'v4_raw_info' in batch:
            for i, p in enumerate(batch['v4_fif_path']):
                if p not in self.raw_info_registry:
                    self.raw_info_registry[p] = batch['v4_raw_info'][i]
        n_samples = len(model_signal_output_unwrapped)
        for i in range(n_samples):
            # invert_reshape_signals returns 2D (C, T_seg) where T_seg = tc * tf.
            mod_in  = model_signal_input_unwrapped[i]        # (C, T_seg)
            mod_out = model_signal_output_unwrapped[i]       # (C, T_seg)
            C, T_seg = mod_out.shape
            tc = T_seg // self.tf
            assert T_seg == tc * self.tf, f"reconstructor expected T_seg ({T_seg}) = tc * tf ({tc}*{self.tf})"

            # Slice this sample's token_dropout out of the packed batch.
            seq_lens     = batch['seq_lens'].cpu().numpy()
            seqlen_accum = int(sum(seq_lens[:i]))
            seqlen       = int(seq_lens[i])
            tok_drop_flat = batch['token_dropout'][seqlen_accum:seqlen_accum + seqlen].cpu().numpy().reshape(-1).astype(bool)
            # Use the PACKED chan_id/t_coarse (same token order as token_dropout) so the
            # (channel, coarse-time) attribution is correct. The *_unwrapped copies are in
            # grid order (invert_reshape_signals reorders them) and would scramble the mask.
            chan_id_flat  = batch['chan_id'][seqlen_accum:seqlen_accum + seqlen].cpu().numpy().reshape(-1).astype(np.int64)
            t_coarse_flat = batch['t_coarse'][seqlen_accum:seqlen_accum + seqlen].cpu().numpy().reshape(-1).astype(np.int64)

            # Scatter (chan_id, t_coarse) -> (C, tc) mask.
            mask_ct = np.zeros((C, tc), dtype=bool)
            valid = (chan_id_flat < C) & (t_coarse_flat < tc)
            mask_ct[chan_id_flat[valid], t_coarse_flat[valid]] = tok_drop_flat[valid]

            # Expand (C, tc) -> (C, T_seg) by repeating each coarse bin tf times.
            mask_full = np.repeat(mask_ct, self.tf, axis=1)   # (C, T_seg)

            # Invert the per-segment normalization back to volts. We build TWO signals
            # per segment — the pure model output (model everywhere) and the hybrid
            # (original on kept cells, model on dropped cells) — plus the dropout mask.
            seg_mean = batch['v4_seg_mean'][i].cpu().numpy().astype(np.float32)    # (C_full,)
            seg_std  = batch['v4_seg_std'][i].cpu().numpy().astype(np.float32)
            if seg_mean.shape[0] != C:
                seg_mean = seg_mean[:C]
                seg_std  = seg_std[:C]
            avg_ref_offset = batch['v4_avg_ref_offset'][i].cpu().numpy().astype(np.float32)  # (T_seg,)
            T_actual = mod_out.shape[1]

            def _add_ref(volts):  # add the per-sample avg-ref offset back (length-robust)
                if avg_ref_offset.shape[0] >= T_actual:
                    return volts + avg_ref_offset[None, :T_actual]
                out = volts.copy()
                out[:, :avg_ref_offset.shape[0]] += avg_ref_offset[None, :]
                return out

            # (1) Pure model output, in volts, for EVERY cell.
            mod_out_volts = _add_ref(
                (mod_out.astype(np.float32) * self.data_norm) * seg_std[:, None] + seg_mean[:, None]
            )

            # (2) "Original" signal used for the unmasked cells of the hybrid. Prefer the
            # raw (resampled, unfiltered) file volts so the hybrid matches the input
            # .fif exactly outside dropped regions; fall back to the model-input
            # (filtered) signal when unfiltered volts weren't shipped.
            if 'v4_unfiltered_volts' in batch:
                orig_volts = batch['v4_unfiltered_volts'][i].cpu().numpy().astype(np.float32)
                if orig_volts.shape[1] >= T_actual:
                    orig_volts = orig_volts[:, :T_actual]
                else:
                    pad = np.full((orig_volts.shape[0], T_actual - orig_volts.shape[1]), np.nan, dtype=np.float32)
                    orig_volts = np.concatenate([orig_volts, pad], axis=1)
                if orig_volts.shape[0] != C:
                    orig_volts = orig_volts[:C]
            else:
                orig_volts = _add_ref(
                    (mod_in.astype(np.float32) * self.data_norm) * seg_std[:, None] + seg_mean[:, None]
                )

            # (3) Hybrid: original on kept cells, model output on dropped cells.
            recon_hybrid = np.where(mask_full, mod_out_volts, orig_volts).astype(np.float32)

            fif_path  = batch['v4_fif_path'][i]
            seg_start = int(batch['v4_seg_start'][i])
            seg_end   = int(batch['v4_seg_end'][i])
            ch_names  = list(batch['v4_channel_names'][i])

            # Buffer: (seg_start, seg_end, model-output, hybrid, dropout-mask, ch_names)
            self.buffer.setdefault(fif_path, []).append(
                (seg_start, seg_end, mod_out_volts, recon_hybrid, mask_full.astype(bool), ch_names)
            )

    @staticmethod
    def _seam_correct_hybrid(hybrid, mask):
        """Re-anchor each infilled (masked) span so it connects to the neighbouring original.

        Per contiguous masked run in a channel:
          - original on BOTH sides -> subtract a linear ramp so the run's ends meet the left/right
            original neighbours (keeps the infill's shape; adds only a linear trend => DC/near-DC
            only, PSD bands untouched);
          - original on ONE side (run touches a recording edge) -> constant DC shift to that side;
          - NEITHER (whole-channel infill, e.g. a bad channel) -> unchanged (no temporal seam).
        Modifies `hybrid` (C, N) in place. `mask` is the bool (C, N) inferred-cell mask.
        """
        C, N = hybrid.shape
        for c in range(C):
            m = mask[c]
            if not m.any() or m.all():
                continue  # nothing masked, or whole channel inferred (no seam to fix)
            d = np.diff(m.astype(np.int8))
            starts = list(np.where(d == 1)[0] + 1)
            ends   = list(np.where(d == -1)[0] + 1)
            if m[0]:
                starts = [0] + starts
            if m[-1]:
                ends = ends + [N]
            for s, e in zip(starts, ends):
                span = hybrid[c, s:e]
                L = hybrid[c, s - 1] if s > 0 else np.nan   # left original neighbour
                R = hybrid[c, e]     if e < N else np.nan   # right original neighbour
                has_L, has_R = (s > 0 and np.isfinite(L)), (e < N and np.isfinite(R))
                mlen = e - s
                if has_L and has_R:
                    dL, dR = L - span[0], R - span[-1]
                    if mlen == 1:
                        hybrid[c, s] = span[0] + 0.5 * (dL + dR)
                    else:
                        hybrid[c, s:e] = span + dL + (dR - dL) * (np.arange(mlen) / (mlen - 1))
                elif has_L:
                    hybrid[c, s:e] = span + (L - span[0])
                elif has_R:
                    hybrid[c, s:e] = span + (R - span[-1])
        return hybrid

    @staticmethod
    def _infill_annotations(full_mask, info):
        """Turn the (C, N) sample-resolution infill mask into MNE annotations describing
        exactly which (channel, time) cells ZUNA reconstructed.

        Returns (annot, fully_infilled_ch_names):
          - Each contiguous masked span becomes a 'ZUNA1.1_infilled' annotation. The label
            deliberately does NOT start with 'BAD', so a subsequent ZUNA run will not treat
            these regions as bad and re-infill already-reconstructed data.
          - A span shared by EVERY channel is emitted once as a global annotation
            (ch_names=[] -> applies to all channels); otherwise ch_names lists the exact
            channels, so channel-specific infills round-trip (see _compute_bad_mask_2d).
          - A channel masked across the WHOLE recording yields one full-duration annotation
            and is returned in `fully_infilled_ch_names` so the caller can drop it from
            info['bads'] (it now carries reconstructed data, so it is no longer "missing").
        """
        from collections import defaultdict
        C, N = full_mask.shape
        sfreq = float(info["sfreq"])
        ch_names = list(info["ch_names"])
        spans = defaultdict(list)          # (start_sample, end_sample) -> [channel idx]
        fully_infilled = set()
        for c in range(C):
            m = full_mask[c]
            if not m.any():
                continue
            if m.all():
                fully_infilled.add(ch_names[c])
            d = np.diff(m.astype(np.int8))
            starts = list(np.where(d == 1)[0] + 1)
            ends   = list(np.where(d == -1)[0] + 1)
            if m[0]:
                starts = [0] + starts
            if m[-1]:
                ends = ends + [N]
            for s, e in zip(starts, ends):
                spans[(s, e)].append(c)

        onsets, durations, descs, ch_lists = [], [], [], []
        for (s, e), chs in sorted(spans.items()):
            onsets.append(s / sfreq)
            durations.append((e - s) / sfreq)
            descs.append("ZUNA1.1_infilled")
            # Empty list = applies to all channels (a global infilled span).
            ch_lists.append([] if len(chs) == C else [ch_names[c] for c in chs])

        annot = mne.Annotations(onset=onsets, duration=durations, description=descs,
                                ch_names=ch_lists, orig_time=None)
        return annot, fully_infilled

    def save_all(self):
        from pathlib import Path
        if not self.buffer:
            print("[v4 recon] no segments buffered — nothing to save.")
            return

        for fif_path, segs in self.buffer.items():
            # Dedup by seg_start (later passes overwrite earlier). Necessary when
            # eeg_eval's make_batch_iterator does multiple epochs over the V4 data.
            seg_by_start = {}
            for s in segs:
                seg_by_start[s[0]] = s
            segs_sorted = sorted(seg_by_start.values(), key=lambda s: s[0])
            info = self.raw_info_registry.get(fif_path)
            if info is None:
                print(f"[v4 recon] no info registered for {fif_path}; skipping.")
                continue

            n_chans = len(info["ch_names"])
            # Determine total length from the largest seg_end
            n_samples_total = max(s[1] for s in segs_sorted)
            # Stitch each segment into full-length arrays. Signals start as NaN (so any
            # uncovered gap is visible); the mask starts as False (uncovered = not dropped).
            full_model  = np.full((n_chans, n_samples_total), np.nan, dtype=np.float32)
            full_hybrid = np.full((n_chans, n_samples_total), np.nan, dtype=np.float32)
            full_mask   = np.zeros((n_chans, n_samples_total), dtype=bool)

            # Map ch_names from the segment back to indices in the full info channel list
            full_ch_to_idx = {name: i for i, name in enumerate(info["ch_names"])}
            for seg_start, seg_end, model_seg, hybrid_seg, mask_seg, ch_names in segs_sorted:
                col_idx = np.array(
                    [full_ch_to_idx[name] for name in ch_names if name in full_ch_to_idx],
                    dtype=np.int64,
                )
                # Segments are (C_seg, T_seg). T_seg may be < (seg_end - seg_start) if the
                # segment was snapped down to a tf multiple. Trim to the common window.
                T_seg = model_seg.shape[1]
                end_clipped = min(seg_end, seg_start + T_seg)
                w    = end_clipped - seg_start
                rows = col_idx[:model_seg.shape[0]]
                nr   = col_idx.shape[0]
                full_model[rows,  seg_start:end_clipped] = model_seg[:nr, :w]
                full_hybrid[rows, seg_start:end_clipped] = hybrid_seg[:nr, :w]
                full_mask[rows,   seg_start:end_clipped] = mask_seg[:nr, :w]

            src_name = Path(fif_path).stem
            # Ensure ends with _raw.fif so MNE doesn't warn
            base = src_name.replace("_raw", "")

            # Describe exactly the cells ZUNA reconstructed (from full_mask) as annotations,
            # and drop any fully-infilled channel from info['bads'] (it now carries recon data).
            out_info = info
            annot = None
            if self.annotate_infill:
                annot, fully_infilled = self._infill_annotations(full_mask, info)
                out_info = info.copy()
                out_info["bads"] = [b for b in out_info.get("bads", []) if b not in fully_infilled]

            def _write(data, path):
                raw_out = mne.io.RawArray(data, out_info, verbose="ERROR")
                if annot is not None and len(annot):
                    raw_out.set_annotations(annot, verbose="ERROR")
                raw_out.save(str(path), overwrite=True, verbose="ERROR")

            # (1) Pure model output everywhere.
            model_path = self.full_dir / f"{base}_raw.fif"
            _write(full_model, model_path)

            # Seam-correct the hybrid so infilled spans connect to the surrounding original.
            if self.seam_correct:
                self._seam_correct_hybrid(full_hybrid, full_mask)

            # (2) Hybrid: original on kept cells, model output on dropped cells.
            hybrid_path = self.hybrid_dir / f"{base}_raw.fif"
            _write(full_hybrid, hybrid_path)

            # The infilled (channel, time) cells are recorded as ZUNA1.1_infilled annotations on
            # the output files above (see _infill_annotations), so no separate mask .npz is written.

            n_dropped = int(full_mask.sum())
            print(f"[v4 recon] wrote {model_path.name}, {hybrid_path.name}  "
                  f"({n_chans} ch, {n_samples_total} samples, {n_dropped} dropped cells)")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


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

        if args.chan_pos_xyz_extremes_type == "fifteens":
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
                if torch.cuda.is_available():
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
            mmap = torch.load(m_path, weights_only=False) # this line was needed ONLY for the Moabb eval datasets (not sure why)


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
            # Testing chop_and_reshape_signals() and invert_reshape_signals() functions with real signals.
            indx = list(range(len(mmap)))
            if self.shuffle:
                random.shuffle(indx)


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

    NOTE: THIS IS DEPRECATED. USE EEGDataset_v2 INSTEAD. FOR NOW, WE CAN JUST STREAM DATASET LOCALLY.
    Iterable dataset that pulls .pt files from Backblaze B2 bucket using boto3 S3-compatible API.
    Modeled after EEGDataset_v2 but with cloud storage integration.
    """
    def __init__(self, args: BCIDatasetArgs):
        print(f"Inside EEGDataset_b2 with B2 bucket: {args.b2_bucket_name}, prefix: {args.data_dir}")
        
        # Validate B2 configuration
        if not all([args.b2_bucket_name, args.b2_endpoint_url, args.b2_access_key_id, args.b2_secret_access_key]):
            raise ValueError("B2 configuration incomplete. Must provide: b2_bucket_name, b2_endpoint_url, b2_access_key_id, b2_secret_access_key")
        
        # Initialize boto3 S3 client for B2
        try:
            import boto3
        except ImportError as error:
            raise ImportError(
                "Backblaze B2 support requires the 'cloud' extra: "
                "pip install 'zuna[cloud]'"
            ) from error
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

        if args.chan_pos_xyz_extremes_type == "fifteens":
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
                if torch.cuda.is_available():
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
        return self # 
        # Unlike STFTProcessor in AY2latent/data_lean.py, nothing to put on device



    @torch.compile() # REINSTATE: commented out for now while working with dropout_chans
    def process(self, eeg_signal, chan_pos, chan_pos_discrete, chan_id, t_coarse, seq_lens, max_tc, token_dropout, pad_mask=None): # freq_masks, # +pad_mask passthrough

        seq_len, channel = eeg_signal.shape # multiple samples packed into single batch
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
            "pad_mask": pad_mask,                       # [N,1] 1=real 0=pad, rides through to the model
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
            'pad_mask':                  torch.vstack([item['pad_mask'] for item in batch[0]]),  # [total,1]
            #
            'max_tc':                   torch.tensor([item['max_tc'] for item in batch[0]]),
            'seq_lens':                 torch.tensor([item['seq_lens'] for item in batch[0]]),
            'ids':                      torch.tensor([item['ids'] for item in batch[0]]),                
            'dataset_id':               torch.tensor([item['dataset_id'] for item in batch[0]]),
        }
        # V4-only reconstruction metadata, detected by 'fif_path' in the segment dicts.
        # Absent for V2/V3/B2 batches, so these keys simply aren't added (backward compatible).
        if batch[0] and ('fif_path' in batch[0][0]):
            packed_batch_dict['v4_seg_mean']       = [item['seg_mean']       for item in batch[0]]
            packed_batch_dict['v4_seg_std']        = [item['seg_std']        for item in batch[0]]
            packed_batch_dict['v4_avg_ref_offset'] = [item['avg_ref_offset'] for item in batch[0]]
            packed_batch_dict['v4_fif_path']       = [item['fif_path']       for item in batch[0]]
            packed_batch_dict['v4_seg_start']      = [item['seg_start']      for item in batch[0]]
            packed_batch_dict['v4_seg_end']        = [item['seg_end']        for item in batch[0]]
            packed_batch_dict['v4_channel_names']  = [item['channel_names']  for item in batch[0]]
            packed_batch_dict['v4_sfreq']          = [item['sfreq']          for item in batch[0]]
            packed_batch_dict['v4_raw_info']       = [item['raw_info']       for item in batch[0]]
            if 'unfiltered_volts' in batch[0][0]:
                packed_batch_dict['v4_unfiltered_volts'] = [item['unfiltered_volts'] for item in batch[0]]
            if 'v4_step_times' in batch[0][0]:
                packed_batch_dict['v4_step_times'] = batch[0][-1]['v4_step_times']
        return packed_batch_dict

    return pack_chans_collate_fn


def create_dataloader_v2(args: BCIDatasetArgs, seed, rank, timeout=200):
    if args.use_v4:
        dataset = EEGDataset_v4(args) # IterableDataset loading .fif files directly for inference!
    elif args.use_v3:
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
        timeout=0 # to pass an assertion error when debugging.


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
