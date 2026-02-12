import torch
from torch.utils.data import DataLoader, IterableDataset
import numpy as np

from dataclasses import dataclass, field
from typing import Union, List, Optional

import torch.distributed as dist
import random
import os

import time
from pathlib import Path

import matplotlib.pyplot as plt


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



@dataclass
class BCIDatasetArgs:
    use_b2: bool = False # If true, use Backblaze B2 for dataset loading (NOT IMPLEMENTED)
    data_dir: str = "/mnt/shared/datasets/bci/"
    export_dir: str = "./output/"  #jm saving pt files - directory to save reconstructed pt files
    glob_filter: str = "**/*.pt" # default is to use all .pt files in all subdirectories.
    chan_num_filter: Union[int, None] = None # None or integer number of channels we want in each sample
    sample_rate: int = 256
    seq_len: int = 1280 
    num_fine_time_pts: int = 128
    use_coarse_time: str = "B" # How to chop signals in to coarse-time, fine-time & channels using chop_and_reshape_signals or chop_signals_only
    cat_chan_xyz_and_eeg: bool = False # alternatively, concatenate channel {x,y,z} and EEG signal in EEGProcessor.process (use in tandem with NoPE)
    dont_noise_chan_xyz: bool = False # If true, do not add noise to channel {x,y,z}-position in EEGProcessor.process (use in tandem with NoPE)
    randomly_permute_sequence: bool = False

    data_norm: float = 1.0 # The norm to divide the data by, to normalize it to [-1,1] range.
    data_clip: float = 1.0 # Clip data to this value after normalization.
    
    sample_duration_seconds: float = 5.0

    num_batches: Union[int, None] = None
    crop_size: Union[int, None] = None

    encoder_input_channels: int = 64 
    decoder_input_channels: int = 64 
    channel_dropout_prob: int | float = -1.0 # Probability of applying channel dropout (negative to turn off)

    batch_size: int = 1 #32 # HARDCODE TO 1. NOT USING. Effective batch size determined by target_packed_seqlen.
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
    masked_in_decoder: bool = True # If true, mask out channels in decoder input when channel is dropped. (Used in training, not eval.)

    num_bins_discretize_xyz_chan_pos: int = 100 # Number of bins to discretize channel positions to use in 4d-RoPE. # 40 with "old" xyz_extremes, 100 with "thirteens" xyz_extremes
    chan_pos_xyz_extremes_type: str = "thirteens" # "old" for v4 dataset or "thirteens" for v5 dataset
    
    # # Backblaze B2 specific fields (for EEGDataset_b2)
    # load_dotenv()
    # b2_bucket_name: Optional[str] = "zyphra-bci" #None # e.g., "zyphra-bci"
    # # JUST USE DATADIR FOR B2 ALSO.  b2_key_prefix: Optional[str] = "datasets/v5/train/" #None  # e.g., "datasets/v5/train/"
    # b2_endpoint_url: Optional[str] = "https://s3.us-west-004.backblazeb2.com" #None  # e.g., "https://s3.us-west-000.backblazeb2.com"
    # b2_access_key_id: Optional[str] = os.getenv("B2_ACCESS_KEY_ID") #None
    # b2_secret_access_key: Optional[str] = os.getenv("B2_SECRET_ACCESS_KEY") #None
    # b2_local_cache_dir: Optional[str] = "/mnt/shared/datasets/bci/b2_cache"  # Local directory to cache downloaded files
    # b2_cache_files: bool = False  # Whether to cache files locally or download on-demand



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
            ## OLD TEST VALUES
            self.xyz_extremes = 1.10*torch.tensor([ 
                [-0.0861, -0.1124, -0.0680], 
                [0.0858, 0.0849, 0.1002]
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

        #JM debug - Check for duplicate files in memmap_paths
        # Disabled verbose output - uncomment for debugging
        # from collections import Counter
        # print(f"\n{'='*80}")
        # print(f"[DATALOADER 🔍] Checking memmap_paths for duplicate files")
        # print(f"{'='*80}")
        # file_basenames = [p.name for p in self.memmap_paths]
        # file_counts = Counter(file_basenames)
        # duplicates = {name: count for name, count in file_counts.items() if count > 1}
        # if duplicates:
        #     print(f"[DATALOADER 🔍] ⚠️  DUPLICATE FILES DETECTED in memmap_paths:")
        #     for name, count in duplicates.items():
        #         print(f"  📂 {name}: appears {count} times")
        # else:
        #     print(f"[DATALOADER 🔍] ✓ No duplicate files in memmap_paths ({len(self.memmap_paths)} unique files)")
        # print(f"{'='*80}\n")
        
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

        #JM debug - Show sharding logic
        # Disabled verbose output
        # print(f"[DATALOADER 🔍] Worker {global_worker_id}/{total_global_workers} assigned {len(sharded_indices_for_this_worker)} files")
        # print(f"  Indices range: {sharded_indices_for_this_worker[:5]}...{sharded_indices_for_this_worker[-5:] if len(sharded_indices_for_this_worker) > 5 else ''}")

        if self.shuffle:
            # 4th. Shuffle the indices assigned to this worker.\
            rng_worker.shuffle(sharded_indices_for_this_worker)
            #JM debug - Show shuffling effect
            # print(f"[DATALOADER 🔍] After shuffle: {sharded_indices_for_this_worker[:5]}...{sharded_indices_for_this_worker[-5:] if len(sharded_indices_for_this_worker) > 5 else ''}")


        # Init for sequence packing
        seqlen_accum = 0
        packed_batch = []

        #JM debug - Track file loads to detect duplicates
        loaded_files = []
        # print(f"[DATALOADER 🔍] Starting to load {len(sharded_indices_for_this_worker)} files...\n")

        # Loop over all the dataset files in this worker's shard.
        for file_load_idx, ids in enumerate(sharded_indices_for_this_worker):
            m_path = self.memmap_paths[int(ids)]

            #JM debug - Track each file load
            loaded_files.append(m_path.name)
            # if file_load_idx < 5 or file_load_idx >= len(sharded_indices_for_this_worker) - 3:
            #     print(f"[DATALOADER 🔍] 📂 Loading file #{file_load_idx}: {m_path.name} (index {ids} in memmap_paths)")

            # mmap = torch.load(m_path) #original line that worked for ALL TRAINING AND EVAL
            mmap = torch.load(m_path, weights_only=False) #jm | this line was needed ONLY for the Moabb eval datasets (not sure why)

            # Handle different dataset structures
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


            # Filter out samples that do not have self.chan_num_filter channels.
            if self.chan_num_filter is not None:
                mmap_filt = []
                chan_pos_filt = []
                chan_pos_discrete_filt = []
                filtered_indices = []  # Track which epochs are kept
                for i in range(len(mmap)):
                    if mmap[i].shape[0]==self.chan_num_filter:
                        mmap_filt.append(mmap[i])
                        chan_pos_filt.append(chan_pos[i])
                        chan_pos_discrete_filt.append(chan_pos_discrete[i])
                    else:
                        filtered_indices.append(i)

                # Debug output
                if len(filtered_indices) > 0:
                    print(f"[DEBUG] Channel filter: expecting {self.chan_num_filter} channels")
                    print(f"[DEBUG] Filtered {len(filtered_indices)}/{len(mmap)} epochs")
                    if len(filtered_indices) <= 20:
                        print(f"[DEBUG] Filtered epoch indices: {filtered_indices}")
                        print(f"[DEBUG] Channel counts: {[mmap[i].shape[0] for i in filtered_indices[:10]]}")

                mmap = mmap_filt
                chan_pos = chan_pos_filt
                chan_pos_discrete = chan_pos_discrete_filt


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


            if False:
                # Dropout scheme used for training:
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
                chan_dropout = []
                for mm in mmap:
                    N = mm.shape[0]
                    if N<=1: # if there is only 1 channel, cannot dropout any.
                        chan_dropout.append([]) # No dropout for this sample.
                        continue
                    M = int(self.channel_dropout_prob * N)
                    random_integers = sorted(random.sample(range(1, N), M))
                    chan_dropout.append(random_integers)


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

            # Flatten list of lists into single list if trying to process each channel as separate sample.
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

            for s in indx:
                #JM CHANGE ORDER
                
                # try:
                # Apply channel dropout here to get boolean mask
                chan_id = reshaped[s][3]
                chan_do = chan_dropout[s]
                dropout_bool = torch.zeros_like(chan_id, dtype=torch.bool)
                for d in chan_do:
                    dropout_bool[chan_id==d] = True

                #jm saving pt files - create sample dict (moved outside if/else to fix sample loss bug)
                sample_dict = {
                    "eeg_signal": eeg_cat[s],
                    "chan_pos": reshaped[s][1],
                    "chan_pos_discrete": reshaped[s][2],
                    "chan_id": reshaped[s][3],
                    "t_coarse": reshaped[s][4],
                    "seq_lens": reshaped[s][5],
                    "chan_dropout": dropout_bool,
                    "ids": ids,
                    "dataset_id": dataset_id,
                    "filename": str(m_path.name),      # Track source filename
                    "sample_idx": s,                    # Track sample index within file
                    "metadata": file_metadata           # Pass through file metadata
                }

                if seqlen_accum < self.target_packed_seqlen:
                    # Batch not full yet, add sample to current batch
                    seqlen_accum += reshaped[s][5]
                    packed_batch.append(sample_dict)
                else:
                    # Batch is full, yield it and start new batch with current sample
                    yield packed_batch
                    #JM CHANGE ORDER
                    packed_batch = [sample_dict]       # Start new batch with current sample
                    seqlen_accum = reshaped[s][5]      # Initialize seqlen with current sample

                import pdb; pdb.set_trace()

                # except Exception as e:
                #     print(f"Error processing sample: {e} : {ids} : {m_path}")
                #     import pdb; pdb.set_trace()
                #     continue

        #JM FIX FIX FIX - Yield any remaining samples in the last partial batch
        # After processing all files, if there are leftover samples in packed_batch
        # that didn't fill up a complete batch, we need to yield them.
        # Without this, the last ~40 samples get lost!
        if len(packed_batch) > 0:
            # print(f"[DATALOADER 🔍] Yielding final partial batch with {len(packed_batch)} samples")
            yield packed_batch

        #JM debug - Summary of loaded files
        # Disabled verbose output
        # print(f"\n{'='*80}")
        # print(f"[DATALOADER 🔍] Summary of files loaded by worker {global_worker_id}")
        # print(f"{'='*80}")
        # loaded_file_counts = Counter(loaded_files)
        # duplicates_loaded = {name: count for name, count in loaded_file_counts.items() if count > 1}
        # if duplicates_loaded:
        #     print(f"[DATALOADER 🔍] ⚠️  FILES LOADED MULTIPLE TIMES by this worker:")
        #     for name, count in duplicates_loaded.items():
        #         print(f"  📂 {name}: loaded {count} times")
        # else:
        #     print(f"[DATALOADER 🔍] ✓ No files loaded multiple times")
        # print(f"[DATALOADER 🔍] Worker {global_worker_id} finished: loaded {len(loaded_files)} files total ({len(set(loaded_files))} unique)")
        # print(f"{'='*80}\n")


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
        if self.diffusion_forcing:
            self.diffusion_forcing_num_frames = args.diffusion_forcing_num_frames



    def to(self, device):
        return self



    @torch.compile()
    def process(self, eeg_signal, chan_pos, chan_pos_discrete, chan_id, t_coarse, seq_lens, chan_dropout):

        seq_len, channel = eeg_signal.shape
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
        print("NOTE: EEGDataset_b2 is not implemented yet. Using EEGDataset_v2 instead.")
        #dataset = EEGDataset_b2(args) # IterableDataset pulling from B2!

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
        timeout=0


    # create sequence packing collator function
    pack_chans_collate_fn = create_pack_chans_collate_fn(args.target_packed_seqlen)


    return DataLoader(
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