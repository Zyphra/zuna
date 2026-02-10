import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
# import zarr
import numpy as np
import math
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




# def chop_signals_only(eeg_signal, chan_pos=None, tf=128):
#     """
#     Just grab the first tf time points from an eeg_signal
#     NOT USING ANYMORE: This is done in chop_and_reshape_signals with  use_coarse_time="B"
#     """
#     num_chans, num_tpts = eeg_signal.shape

#     eeg_reshaped = eeg_signal[:, :tf]  # just grab the first tf time points
#     chan_pos_reshaped = chan_pos
#     tc_reshaped = torch.zeros(num_chans,1)
#     chan_id_reshaped = torch.arange(num_chans).unsqueeze(-1)

#     # eeg_reshaped.shape = [num_chans, tf]
#     # chan_pos_reshaped.shape = [num_chans, 3]
#     # tc_reshaped.shape = [num_chans, 1] <- zeros
#     # num_chans = int
#     return eeg_reshaped, chan_pos_reshaped, chan_id_reshaped, tc_reshaped, num_chans


def chop_and_reshape_signals(eeg_signal, chan_pos=None, chan_pos_discrete=None, chan_dropout=None, tf=128, use_coarse_time="B"):
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
        assert num_tpts%tf==0
        tc = num_tpts//tf

    # print(f"Inside chop_and_reshape_signals,")
    # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

    if use_coarse_time=="A":
        # Keep same coarse-time values together in reshaping.
        seqlen = num_chans*tc
        eeg_reshaped = eeg_signal.reshape(num_chans, tc, tf).transpose(0,1).reshape(seqlen,tf)
        chan_pos_reshaped = chan_pos.repeat((tc,1)) if chan_pos is not None else None
        chan_pos_discrete_reshaped = chan_pos_discrete.repeat((tc,1)) if chan_pos_discrete is not None else None
        chan_id_reshaped = torch.arange(num_chans).unsqueeze(-1).repeat((tc,1))
        tc_reshaped = torch.arange(tc).repeat((num_chans,1)).T.reshape(seqlen,1)

    elif use_coarse_time=="B" or use_coarse_time=="D":
        # Keep same channels together in reshaping
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
    return eeg_reshaped, chan_pos_reshaped, chan_pos_discrete_reshaped, chan_id_reshaped, tc_reshaped, seqlen




def invert_reshape_signals(sig_reshaped, pos_reshaped=None, pos_discrete_reshaped=None, id_reshaped=None, tc_reshaped=None, num_chans=62, tf=128, use_coarse_time="B"):
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

    tc = sig_reshaped.shape[0]//num_chans
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



# @dataclass
# class BCIDatasetArgs_old:
#     data_path: str = "/mnt/shared/datasets/eeg_data.dat"
#     sample_rate: int = 256
#     sample_duration_seconds: float = 5.0

#     memmap_shape: List[int] = field(default_factory=lambda: [992252, 64, 1280]) # was for /workspace/bci/data/mmap/mbad0_992252_64_1280.mmap
#     memmap_dtype: str = "torch.bfloat16" # "float64" # "float32" # 
#     crop_size: Union[int, None] = None

#     encoder_input_channels: int = 64
#     decoder_input_channels: int = 64

#     batch_size: int = 32
#     num_workers: int = 8
#     pin_memory: bool = True
#     shuffle: bool = True
#     persistent_workers: bool = True
#     prefetch_factor: Union[int, None] = 2

#     diffusion_forcing: bool = False
#     diffusion_noise_schedule: str = "linear"
#     diffusion_forcing_num_frames: int = 1

#     patching_type: str = "frames"
#     stft_global_sigma: Union[str, float] = 1.0

#     seq_len: int = 1280


@dataclass
class BCIDatasetArgs:
    use_b2: bool = False # If true, use Backblaze B2 for dataset loading, otherwise use local filesystem.
    # data_paths: list[str] = field(default_factory=lambda: ["/mnt/shared/datasets/eeg_data.dat"])
    data_dir: str = "/mnt/shared/datasets/bci/"
    export_dir: str = "./output/"  #jm saving pt files - directory to save reconstructed pt files
    glob_filter: str = "**/*.pt" # default is to use all .pt files in all subdirectories.
    chan_num_filter: Union[int, None] = None # None or integer number of channels we want in each sample
    sample_rate: int = 256 # 512 # Passing in from config now.
    seq_len: int = 1280 # 2560 # Passing in from config now.
    num_fine_time_pts: int = 128
    use_coarse_time: str = "B" # How to chop signals in to coarse-time, fine-time & channels using chop_and_reshape_signals or chop_signals_only
    cat_chan_xyz_and_eeg: bool = True
    dont_noise_chan_xyz: bool = False # If true, do not add noise to channel {x,y,z}-position in EEGProcessor.process (use in tandem with NoPE)
    randomly_permute_sequence: bool = False

    # data_norm: int = 1 #5 # 1 # Passing in from config now. (CW) - this is the norm to divide the data by, to normalize it to [-1,1] range.
    # num_chans: int = 1 #23 # 64 # (CW) - NOT USED YET! (BUT SHOULD IMPLEMENT IT)
    sample_duration_seconds: float = 5.0

    num_batches: Union[int, None] = None

    # memmap_shape: List[int] = field(default_factory=lambda: [992252, 64, 1280]) # was for /workspace/bci/data/mmap/mbad0_992252_64_1280.mmap
    # memmap_dtype: str = "float32" #  "torch.bfloat16" # "float64" # (CW) - Think this is unused...
    crop_size: Union[int, None] = None

    encoder_input_channels: int = 64 # NOT USING ANYLONGER. GET RID OF.
    decoder_input_channels: int = 64 # NOT USING ANYLONGER. GET RID OF.
    channel_dropout_prob: int | float = -1.0 # Probability of applying channel dropout (negative to turn off)

    batch_size: int = 32
    target_packed_seqlen: int =  16384
    do_N_epochs: Union[int, None] = None
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: Union[int, None] = 2
    shuffle: bool = True
    seed: Union[int, None] = None

    diffusion_forcing: bool = False
    diffusion_noise_schedule: str = "linear"
    diffusion_forcing_num_frames: int = 1

    patching_type: str = "frames"
    stft_global_sigma: Union[str, float] = 1.0
    masked_in_decoder: bool = True # If true, mask out channels in decoder input when channel is dropped. (true works, false does not)

    num_bins_discretize_xyz_chan_pos: int = 100 # Number of bins to discretize channel positions to use in 4d-RoPE. # 40 with "old" xyz_extremes, 100 with "thirteens" xyz_extremes
    chan_pos_xyz_extremes_type: str = "thirteens" # "old" for v4 dataset or "thirteens" for v5 dataset
    
    # Backblaze B2 specific fields (for EEGDataset_b2)
    load_dotenv()
    b2_bucket_name: Optional[str] = "zyphra-bci" #None # e.g., "zyphra-bci"
    # JUST USE DATADIR FOR B2 ALSO.  b2_key_prefix: Optional[str] = "datasets/v5/train/" #None  # e.g., "datasets/v5/train/"
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

    # print(f"Inside discretize_chan_pos:")
    # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

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


class EEGDataset_v2(IterableDataset):
    """
    Iterable dataset because we have lots more data for training.
    """
    def __init__(self, args: BCIDatasetArgs):
        # print(f"{args=}")

        print(f"Inside EEGDataset_v2 with {args.glob_filter=}")
        self.memmap_paths = list(Path(args.data_dir).glob(args.glob_filter))
        self.shuffle = args.shuffle
        self.seed = args.seed
        self.num_workers = args.num_workers 
        self.output_channels = args.decoder_input_channels
        self._current_epoch = 0 # To be updated by the training loop
        self.num_fine_time_pts = args.num_fine_time_pts
        self.use_coarse_time = args.use_coarse_time
        self.cat_chan_xyz_and_eeg = args.cat_chan_xyz_and_eeg
        self.target_packed_seqlen = args.target_packed_seqlen
        self.do_N_epochs = args.do_N_epochs
        self.glob_filter = args.glob_filter
        self.chan_num_filter = args.chan_num_filter
        self.randomly_permute_sequence = args.randomly_permute_sequence
        self.channel_dropout_prob = args.channel_dropout_prob
        self.num_bins = args.num_bins_discretize_xyz_chan_pos

        if args.chan_pos_xyz_extremes_type == "old":
            ## OLD TEST VALUES: (CW - WHAT I WAS USING PRIOR TO TEST104 and new v5 dataset)
            self.xyz_extremes = 1.10*torch.tensor([ 
                [-0.0861, -0.1124, -0.0680], 
                [0.0858, 0.0849, 0.1002]
            ])
            # num_bins = 40 #10

        elif args.chan_pos_xyz_extremes_type == "thirteens":
            ##PICK WORKING VALUES BY EYE BALLING. (CW - USING THESE FOR TO TEST104 and new v5 dataset)
            self.xyz_extremes = torch.tensor([ 
                [-0.13, -0.13, -0.13], 
                [ 0.13,  0.13,  0.13]
            ])
            # num_bins = 100

        elif args.chan_pos_xyz_extremes_type == "twelves":
            ##PICK WORKING VALUES BY EYE BALLING. (CW - USING THESE FOR bigrun15 and new v5 dataset)
            self.xyz_extremes = torch.tensor([ 
                [-0.12, -0.12, -0.12], 
                [ 0.12,  0.12,  0.12]
            ])
            # num_bins = 100

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
        # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

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
            # print("SHUFFLING DATASET!", end=" ")
            # 1st. Set different deterministic random seeds for each rank and worker.    
            if self.seed is not None:
                # print("SEED NOT NONE!")
                base_seed = int(self.seed + (1e15 * self._current_epoch))
                rng_base = random.Random(base_seed)
                #print(f"{base_seed=}, {rng_base=}")
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
                # print("SEED IS NONE!")
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
            # mmap = torch.load(m_path) #original line that worked for ALL TRAINING AND EVAL 
            mmap = torch.load(m_path, weights_only=False) #jm | this line was needed ONLY for the Moabb eval datasets (not sure why)

            # print(f"Inside EEGDataset_v2.__iter__, after loading memmap file.")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3) 

            # Handle different dataset structures
            #jm saving pt files - extract and store metadata
            if isinstance(mmap,dict):
                num_samps = len(mmap['data'])
                chan_pos = mmap['channel_positions']
                file_metadata = mmap.get('metadata', {})  # Get metadata for this file
                mmap = mmap['data']
            else: # assuming mmap is a tensor
                num_samps, num_chans, num_t = mmap.shape
                chan_pos = [torch.zeros(num_chans,3) for i in range(num_samps)]     # list of dummy channel positions (all-zeros).
                file_metadata = {}  # Empty metadata for tensor format
                mmap = list(torch.unbind(mmap, dim=0))                              # turn 3D-tensor into list of tensors.

            chan_pos_discrete = [discretize_chan_pos(cp, self.xyz_extremes, self.num_bins) for cp in chan_pos]

            # # Sanity check 1: printing discetization of channel position
            # for c in range(21):
            #     print(f"{chan_pos[0][c]} --> {chan_pos_discrete[0][c]}") 


            # # Sanity check 2: Ensure unique discrete positions match unique continuous positions
            # cp = chan_pos[0].cpu().numpy()
            # cpd = chan_pos_discrete[0].cpu().numpy()
            # assert np.unique(cpd, axis=0).shape == np.unique(cp, axis=0).shape, \
            #     f"Discretization error: unique discrete positions shape {np.unique(cpd, axis=0).shape} != unique continuous positions shape {np.unique(cp, axis=0).shape} with {num_bins=}."


            # Sanity check 3: 3D scatter plot of channel positions and discretized positions
            plot_chan_pos_comparison = False
            if plot_chan_pos_comparison:
                # from mpl_toolkits.mplot3d import Axes3D

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


            # print(f"Inside EEGDataset_v2.__iter__, before filtering and reshaping.")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

            # (CW) - Filter out samples that do not have self.chan_num_filter channels. This is pretty quick - not the source of data_t slowdown
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
                # print(f"Filtering out samples that do not have 21 channels. {len(mmap_filt)} remain.")
                #jm saving pt files - Note: metadata is per-file, not per-sample, so no filtering needed


            # Shuffle the channels randomly to see if the model can still learn from concat'd {x,y,z}-position or RoPE on discretized xyz positions
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



            # Plot channel locations for several downsampling levels. (for super-resolution evals datasets only)
            if False:
                fig, axes = plt.subplots(1, 1, figsize=(12, 12), subplot_kw=dict(projection="3d"))
                data = chan_pos[0]
                numm_channs = data.shape[0]
                # Get max/min for each axis, across data & reconst for subplot consistency
                max_x = 1.1*data[:,0].max()
                min_x = 1.1*data[:,0].min()
                max_y = 1.1*data[:,1].max()
                min_y = 1.1*data[:,1].min()
                max_z = 1.1*data[:,2].max()
                min_z = 1.1*data[:,2].min()

                axes.view_init(elev=20, azim=120)
                axes.set_box_aspect([1, 1, 1])
                axes.set_xlim(min_x, max_x)
                axes.set_ylim(min_y, max_y)
                axes.set_zlim(min_z, max_z)
                #

                # Scatter full resolution data.
                axes.scatter(data[:, 0], data[:, 1], data[:, 2], marker='o', s=60, facecolors='none', edgecolors='r', alpha=1)

                # Scatter 2x downsampled data.
                axes.scatter(data[::2, 0], data[::2, 1], data[::2, 2], marker='x', s=100, facecolors='b', edgecolors='b', alpha=1)
                
                # Scatter 4x downsampled data.
                axes.scatter(data[::4, 0], data[::4, 1], data[::4, 2], marker='+', s=100, facecolors='g', edgecolors='g', alpha=1)
                
                # Scatter 8x downsampled data.
                axes.scatter(data[::8, 0], data[::8, 1], data[::8, 2], marker='d', s=100, facecolors='y', edgecolors='y', alpha=1)

                # Scatter 16x downsampled data.
                axes.scatter(data[::16, 0], data[::16, 1], data[::16, 2], marker='s', s=100, facecolors='c', edgecolors='c', alpha=1)

                axes.set_xlabel('X')
                axes.set_ylabel('Y')
                axes.set_zlabel('Z')

                plt.legend([f"full res ({numm_channs}ch)", f"scaled 2x ({numm_channs//2}ch)", f"scaled 4x ({numm_channs//4}ch)", f"scaled 8x ({numm_channs//8}ch)", f"scaled 16x ({numm_channs//16}ch)"], loc='upper left')
                plt.tight_layout(rect=[0, 0, 1, 0.95]) # leave some space at top for suptitle
                plt.suptitle(f"EEG position downsampling {self.memmap_paths[0].parts[5]}", fontsize=16, fontweight='bold')

                plt.savefig(f"figures/eeg_position_downsampling_{self.memmap_paths[0].parts[5]}.png", dpi=300, bbox_inches='tight')
                plt.close()

                print(f"\n\nMean +/- Std (x,y,z) of channel positions in {self.memmap_paths[0].parts[5]} ({numm_channs}ch) at different downsampling levels:")
                print(f"\nFull res : ({data[:,0].mean(): 0.4f} +/- {data[:,0].std(): 0.4f}), ({data[:,1].mean(): 0.4f} +/- {data[:,1].std(): 0.4f}), ({data[:,2].mean(): 0.4f} +/- {data[:,2].std(): 0.4f})")
                print(f"Scaled 2x: ({data[::2,0].mean(): 0.4f} +/- {data[::2,0].std(): 0.4f}), ({data[::2,1].mean(): 0.4f} +/- {data[::2,1].std(): 0.4f}), ({data[::2,2].mean(): 0.4f} +/- {data[::2,2].std(): 0.4f})")
                print(f"Scaled 4x: ({data[::4,0].mean(): 0.4f} +/- {data[::4,0].std(): 0.4f}), ({data[::4,1].mean(): 0.4f} +/- {data[::4,1].std(): 0.4f}), ({data[::4,2].mean(): 0.4f} +/- {data[::4,2].std(): 0.4f})")
                print(f"Scaled 8x: ({data[::8,0].mean(): 0.4f} +/- {data[::8,0].std(): 0.4f}), ({data[::8,1].mean(): 0.4f} +/- {data[::8,1].std(): 0.4f}), ({data[::8,2].mean(): 0.4f} +/- {data[::8,2].std(): 0.4f})")
                print(f"Scaled 16x: ({data[::16,0].mean(): 0.4f} +/- {data[::16,0].std(): 0.4f}), ({data[::16,1].mean(): 0.4f} +/- {data[::16,1].std(): 0.4f}), ({data[::16,2].mean(): 0.4f} +/- {data[::16,2].std(): 0.4f})")


                print(f"Inside EEGDataset_v2.__iter__, before channel dropout.")
                import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)







            if False:
                ## NOTE: THIS WAS OUR FIRST DROPOUT SCHEME USED FOR TRAINING - FOR TEST69 TO TEST83
                # Apply channel dropout right here to get list of channels to drop
                chan_dropout = []
                for mm in mmap:
                    if random.random() < self.channel_dropout_prob:
                        N = mm.shape[0]
                        if N<=1: # if there is only 1 channel, cannot dropout any.
                            chan_dropout.append([]) # No dropout for this sample.
                            continue
                        M = random.randint(1, N-1)
                        random_integers = sorted(random.sample(range(1, N), M))
                        chan_dropout.append(random_integers)
                    else:
                        chan_dropout.append([]) # No dropout for this sample.


            if False:
                ## NOTE: USING THIS IMPROVED DROPOUT SCHEME USED FOR TRAINING - STARTING WITH TEST84 - TRYING OUT THERE.
                # Apply NEW channel dropout right here to get list of channels to drop
                #   a. self.channel_dropout_prob determines whether we do channel dropout for this sample.
                #   If we do channel dropout, 
                #       b. with p=0.8, we drop between 1 and N/2 chans with uniform probability.
                #       c. with p=0.2, we drop between N/2 and N-1 chans with uniform probability.
                chan_dropout = []
                for mm in mmap:
                    if random.random() < self.channel_dropout_prob:
                        N = mm.shape[0]
                        if N<=1: # if there is only 1 channel, cannot dropout any.
                            chan_dropout.append([]) # No dropout for this sample.
                            continue
                        if random.random() < 0.8:
                            M = random.randint(1, N//2)
                        else:
                            M = random.randint(N//2, N-1)
                        random_integers = sorted(random.sample(range(1, N), M))
                        chan_dropout.append(random_integers)
                    else:
                        chan_dropout.append([]) # No dropout for this sample.


            if True:
                ## NOTE: THIS FIXED DROPOUT RATE SCHEME USED FOR EVALS. FIRST, RANDOMLY DROP p*N CHANNELS.
                #        CAN ALSO DROP OUT CHANNELS IN AN ORGANIZED WAY FROM THE GRID.
                chan_dropout = []
                for mm in mmap:
                    N = mm.shape[0]
                    if N<=1: # if there is only 1 channel, cannot dropout any.
                        chan_dropout.append([]) # No dropout for this sample.
                        continue
                    M = int(self.channel_dropout_prob * N)
                    random_integers = sorted(random.sample(range(1, N), M))
                    chan_dropout.append(random_integers)

                # print(f"Inside EEGDataset_v2.__iter__, after channel dropout.")
                # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

                


            if False:
                # Grid-based channel dropout scheme - to match what I think papers are doing - on BerlinBCI and BCI2000 datasets.
                assert isinstance(self.channel_dropout_prob, int), f"channel_dropout_prob must be an integer for grid-based dropout, got {self.channel_dropout_prob}"
                N = mmap[0].shape[0] # number of channels in each sample
                M = len(mmap) # number of samples
                chan_dropout = [ sorted(set(range(0, N)) - set(range(0, N, self.channel_dropout_prob))) ] * M



            if False:
                # Grid-based channel dropout scheme - to match what I think papers are doing - on LocalizeMI dataset
                from utils_pt_mne import egi_montage_subsampling
                chan_dropout = []
                for ii, mm in enumerate(mmap):
                    N = mm.shape[0]
                    chan_dropout.append(egi_montage_subsampling(montage=N//self.channel_dropout_prob, subject_coords=chan_pos[ii]))




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
                reshaped = [chop_and_reshape_signals(m, c, cd, do, self.num_fine_time_pts, self.use_coarse_time) for m,c,cd,do in zip(mmap, chan_pos, chan_pos_discrete, chan_dropout)]
            else:
                print(f"Dont understand {self.use_coarse_time=}")


            # print(f"Inside EEGDataset_v2.__iter__, after chop_and_reshape_signals with {self.use_coarse_time=}.")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3) 

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
                                                          use_coarse_time=self.use_coarse_time,
                    )
                else:
                    xxx, _, _, _, _ = invert_reshape_signals(sig_reshaped=eeg_cat[indx0], 
                                                          pos_reshaped=reshaped[indx0][1],
                                                          num_chans=num_chans, 
                                                          tf=tf,
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
            # print(f"{rank=} : {ids=} : {m_path}")

            # print(f"At the end of EEGDataset_v2.__iter__, pass out dropout?")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

            for s in indx:
                try:
                    if seqlen_accum < self.target_packed_seqlen:
                        # Collect up samples in packed_batch until seqlen_accum > self.target_seqlen
                        seqlen_accum += reshaped[s][5]

                        # Apply channel dropout here to get boolean mask
                        chan_id = reshaped[s][3]
                        chan_do = chan_dropout[s]
                        dropout_bool = torch.zeros_like(chan_id, dtype=torch.bool)
                        for d in chan_do:
                            dropout_bool[chan_id==d] = True

                        #jm saving pt files - add tracking fields for metadata and file reconstruction
                        packed_batch.append(
                            {"eeg_signal": eeg_cat[s],
                            "chan_pos": reshaped[s][1],
                            "chan_pos_discrete": reshaped[s][2],
                            "chan_id": reshaped[s][3],
                            "t_coarse":reshaped[s][4],
                            "seq_lens":reshaped[s][5],
                            "chan_dropout": dropout_bool,
                            "ids": ids,
                            "dataset_id": dataset_id,
                            "filename": str(m_path.name),      # Track source filename
                            "sample_idx": s,                    # Track sample index within file
                            "metadata": file_metadata}          # Pass through file metadata
                        )
                    else:
                        # Then yield packed_batch and reset list to []
                        yield packed_batch
                        seqlen_accum = 0
                        packed_batch = []

                except Exception as e:
                    print(f"Error processing sample: {e} : {ids} : {m_path}")
                    continue


class EEGDataset_b2(IterableDataset):
    """
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
        self.randomly_permute_sequence = args.randomly_permute_sequence
        self.channel_dropout_prob = args.channel_dropout_prob
        self.num_bins = args.num_bins_discretize_xyz_chan_pos

        if args.chan_pos_xyz_extremes_type == "old":
            ## OLD TEST VALUES: (CW - WHAT I WAS USING PRIOR TO TEST104 and new v5 dataset)
            self.xyz_extremes = 1.10*torch.tensor([ 
                [-0.0861, -0.1124, -0.0680], 
                [0.0858, 0.0849, 0.1002]
            ])
            # num_bins = 40 #10
        elif args.chan_pos_xyz_extremes_type == "thirteens":
            ##PICK WORKING VALUES BY EYE BALLING. (CW - USING THESE FOR TO TEST104 and new v5 dataset)
            self.xyz_extremes = torch.tensor([ 
                [-0.13, -0.13, -0.13], 
                [ 0.13,  0.13,  0.13]
            ])
            # num_bins = 100
        elif args.chan_pos_xyz_extremes_type == "twelves":
            ##PICK WORKING VALUES BY EYE BALLING. (CW - USING THESE FOR TO TEST104 and new v5 dataset)
            self.xyz_extremes = torch.tensor([ 
                [-0.12, -0.12, -0.12], 
                [ 0.12,  0.12,  0.12]
            ])
            # num_bins = 100

        else:
            raise ValueError(f"Invalid value for args.chan_pos_xyz_extremes_type: {args.chan_pos_xyz_extremes_type} - must be one of 'old', 'thirteens', 'twelves'.")
        
        # List all .pt files in the B2 bucket/prefix
        print(f"Listing .pt files in B2 bucket: {self.bucket_name}, prefix: {self.key_prefix}.  Will take a few mins...")        
        
        # print(f"In Iterable EEGDataset_b2.__init__, ")
        # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

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
        # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)
    
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
            
            # Randomly permute sequence if enabled
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
            
            # Apply channel dropout (same as EEGDataset_v2)
            if False:
                # Apply chan dropout with channel_dropout_prob and then 80% chance of dropping out less than half the channels, 20% chance of dropping out more than half the channels.
                chan_dropout = []
                for mm in mmap:
                    if random.random() < self.channel_dropout_prob:
                        N = mm.shape[0]
                        if N <= 1:
                            chan_dropout.append([])
                            continue
                        if random.random() < 0.8:
                            M = random.randint(1, N // 2)
                        else:
                            M = random.randint(N // 2, N - 1)
                        random_integers = sorted(random.sample(range(1, N), M))
                        chan_dropout.append(random_integers)
                    else:
                        chan_dropout.append([])


            if True:
                ## NOTE: THIS FIXED DROPOUT RATE SCHEME USED FOR EVALS. FIRST, RANDOMLY DROP p*N CHANNELS.
                #        CAN ALSO DROP OUT CHANNELS IN AN ORGANIZED WAY FROM THE GRID.
                chan_dropout = []
                for mm in mmap:
                    N = mm.shape[0]
                    if N<=1: # if there is only 1 channel, cannot dropout any.
                        chan_dropout.append([]) # No dropout for this sample.
                        continue
                    M = int(self.channel_dropout_prob * N)
                    random_integers = sorted(random.sample(range(1, N), M))
                    chan_dropout.append(random_integers)
            
            # Shuffle samples within file
            indx = list(range(len(mmap)))
            if self.shuffle:
                random.shuffle(indx)
            
            # Reshape signals
            if self.use_coarse_time in {"A", "B", "C", "D"}:
                reshaped = [chop_and_reshape_signals(m, c, cd, do, self.num_fine_time_pts, self.use_coarse_time) 
                           for m, c, cd, do in zip(mmap, chan_pos, chan_pos_discrete, chan_dropout)]
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

            # print(f"In EEGDataset_b2.__iter__, after loading and processing from B2: {b2_key=}")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)
            
            # Yield packed batches
            for s in indx:
                try:
                    if seqlen_accum < self.target_packed_seqlen:
                        seqlen_accum += reshaped[s][5]
                        
                        # Apply channel dropout boolean mask
                        chan_id = reshaped[s][3]
                        chan_do = chan_dropout[s]
                        dropout_bool = torch.zeros_like(chan_id, dtype=torch.bool)
                        for d in chan_do:
                            dropout_bool[chan_id == d] = True
                        
                        packed_batch.append({
                            "eeg_signal": eeg_cat[s],
                            "chan_pos": reshaped[s][1],
                            "chan_pos_discrete": reshaped[s][2],
                            "chan_id": reshaped[s][3],
                            "t_coarse": reshaped[s][4],
                            "seq_lens": reshaped[s][5],
                            "chan_dropout": dropout_bool,
                            "ids": ids,
                            "dataset_id": dataset_id
                        })
                    else:
                        yield packed_batch
                        seqlen_accum = 0
                        packed_batch = []
                
                except Exception as e:
                    print(f"Error processing sample: {e} : {ids} : {b2_key}")
                    continue


def beta_sched(t_shape, device, dtype):
    t = torch.randn(t_shape, device=device, dtype=dtype) * 2 + 0.3
    t = torch.sigmoid_(t) * 1.02 - 0.01
    return t.clamp_(0,1)


class EEGProcessor:
    def __init__(self, args: BCIDatasetArgs):
        # self.args = args
        self.diffusion_noise_schedule = args.diffusion_noise_schedule
        self.global_sigma = args.stft_global_sigma
        self.patch_type = args.patching_type
        self.diffusion_forcing = args.diffusion_forcing
        self.cat_chan_xyz_and_eeg = args.cat_chan_xyz_and_eeg
        self.dont_noise_chan_xyz = args.dont_noise_chan_xyz
        self.masked_in_decoder = args.masked_in_decoder
        # self.channel_dropout_prob = args.channel_dropout_prob
        if self.diffusion_forcing:
            self.diffusion_forcing_num_frames = args.diffusion_forcing_num_frames
            # assert args.seq_len % self.diffusion_forcing_num_frames == 0, (
            #     "Diffusion forcing num frames must be divisible by seq_len"
            # )


    def to(self, device):
        return self # (CW)
        # pass (was this)
        # Unlike STFTProcessor in AY2latent/data_lean.py, nothing to put on device



    # @torch.compile() # (CW) - REINSTATE: commented out for now while working with dropout_chans
    def process(self, eeg_signal, chan_pos, chan_pos_discrete, chan_id, t_coarse, seq_lens, chan_dropout): # freq_masks,

        # batch, seq_len, channel = eeg_signal.shape
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

        # if diffusion forcing, duplicate dim 1 to match decoder_stft seq_len such that t1 t2 t3 -> t1 t1 ... t2 t2 ... t3 t3 ..
        if self.diffusion_forcing:
            t = torch.repeat_interleave(t, self.diffusion_forcing_num_frames, dim=1)[:, :seq_len, :]

        sigma = self.global_sigma

        # print(f"Before dropout_chans, ...")
        # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

        # Apply channel dropout here to eeg_signal
        eeg_signal_masked = eeg_signal.clone()
        eeg_signal_masked[chan_dropout.squeeze(-1),:] = 0.0

        # Make random noise signal. But, maintain x,y,z channel positions if you concated them in.
        noise = torch.randn_like(eeg_signal) * sigma
        if self.dont_noise_chan_xyz:
            if self.cat_chan_xyz_and_eeg:
                noise[:,:3] = eeg_signal[:,:3] # dont add noise to {x,y,z}-position channels.   
                eeg_signal_masked[:,:3] = eeg_signal[:,:3] # dont mask {x,y,z}-position channels.
            else:
                print("NOTE: EEG channel {x,y,z}-position was never concatenated into signal.")
                import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

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

            dropout_mask = chan_dropout.squeeze(-1)

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


            # print(f"INside EEGProcessor.process, plotting sample of noisy and clean signals.")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

            print(
                eeg_signal, 
                chan_pos, 
                chan_pos_discrete, 
                chan_id, 
                t_coarse, 
                seq_lens, 
                chan_dropout,
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
            "eeg_signal": eeg_signal,   # just passing eeg_signal through.
            "chan_pos": chan_pos,         # just passing chan_pos through.
            "chan_pos_discrete": chan_pos_discrete,         # just passing chan_pos_discrete through.
            "chan_id": chan_id,           # just passing chan_id through.
            "seq_lens": seq_lens,         # just passing seq_lens through.
            "t_coarse": t_coarse,         # just passing t_coarse through.
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
        # print(f"Worker {worker_id} on rank {rank} using seed {worker_seed}")


def create_pack_chans_collate_fn(target_packed_seqlen=1): #batch,
    """
    Do Sequence packing here and in EEGDataset_v2
    """
    def pack_chans_collate_fn(batch):
        #jm saving pt files - include tracking fields for metadata
        packed_batch_dict = {
            'eeg_signal':               torch.vstack([item['eeg_signal'] for item in batch[0]]),
            'chan_pos':                 torch.vstack([item['chan_pos'] for item in batch[0]]),
            'chan_pos_discrete':        torch.vstack([item['chan_pos_discrete'] for item in batch[0]]),
            'chan_id':                  torch.vstack([item['chan_id'] for item in batch[0]]),
            't_coarse':                 torch.vstack([item['t_coarse'] for item in batch[0]]),
            'chan_dropout':             torch.vstack([item['chan_dropout'] for item in batch[0]]),
            #
            'seq_lens':                 torch.tensor([item['seq_lens'] for item in batch[0]]),
            'ids':                      torch.tensor([item['ids'] for item in batch[0]]),
            'dataset_id':               torch.tensor([item['dataset_id'] for item in batch[0]]),
            'filename':                 [item['filename'] for item in batch[0]],      # List of filenames
            'sample_idx':               [item['sample_idx'] for item in batch[0]],    # List of sample indices
            'metadata':                 [item['metadata'] for item in batch[0]],      # List of metadata dicts
        }
        return packed_batch_dict

    return pack_chans_collate_fn


def create_dataloader_v2(args: BCIDatasetArgs, seed, rank, timeout=200):
    if args.use_b2:
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

    # print(f"Inside create_dataloader_v2")
    # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

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


# def create_dataloader(args: BCIDatasetArgs, seed, rank, timeout=200):
#     dataset = EEGDataset(args) # (CW) - Use non-iterable dataset for eval datasets.

#     # print(f"Inside create_dataloader in eeg_data.py, {dataset=}") # (CW)

#     is_distributed = dist.is_available() and dist.is_initialized()
#     sampler = None
#     shuffle = args.shuffle  # Keep original shuffle intent if not distributed

#     # print(f"{is_distributed=}")

#     if is_distributed:
#         world_size = dist.get_world_size()
#         global_rank = dist.get_rank()  # Use global rank for sampler
#         sampler = DistributedSampler(     # (CW) - no sampler or shuffle with IterableDataset 
#             dataset,
#             num_replicas=world_size,
#             rank=global_rank,
#             shuffle=args.shuffle,  # Sampler handles shuffling
#             seed=seed,  # Ensure consistent shuffle across epochs/resumption
#         )
#         shuffle = False  # Sampler handles shuffling, so DataLoader shouldn't
#         print(f"Rank {global_rank}/{world_size}: Using DistributedSampler.")
#         sampler.set_epoch(0)  # (CW) - no sampler with IterableDataset 

#     import functools

#     init_fn = functools.partial(worker_init_fn, seed=seed, rank=rank)

#     # print(f"Inside create_dataloader in eeg_data.py, {is_distributed=}") # (CW)

#     if args.num_workers==0:
#         timeout=0 # (CW) - to pass an assertion error when debugging.

#     return torch.utils.data.DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         pin_memory=args.pin_memory,
#         shuffle=shuffle,  # (CW) - no shuffle with IterableDataset
#         sampler=sampler,   # (CW) - no sampler with IterableDataset
#         persistent_workers=args.persistent_workers,
#         prefetch_factor=args.prefetch_factor,
#         worker_init_fn=init_fn,
#         drop_last=is_distributed,
#         timeout=timeout,
#         in_order=False,
#     )