# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


# 1st, setup tmux and docker with lingua.sh
#   >> bash /data/home/chris/workspace/AY2l/lingua/lingua.sh # (on VP)
#
# 2nd, run something like:
#   >> CUDA_VISIBLE_DEVICES=0 python3 apps/AY2latent_bci/train_compile_fwd.py config=apps/AY2latent_bci/configs/config_bci.yaml 
#   >> CUDA_VISIBLE_DEVICES=1 python3 apps/AY2latent_bci/train_compile_fwd.py config=apps/AY2latent_bci/configs/config_bci.yaml 
#   >> CUDA_VISIBLE_DEVICES=2 python3 apps/AY2latent_bci/train_compile_fwd.py config=apps/AY2latent_bci/configs/config_bci.yaml 
#   >> CUDA_VISIBLE_DEVICES=5 python3 apps/AY2latent_bci/train_compile_fwd.py config=apps/AY2latent_bci/configs/config_bci.yaml 
#       (OR)
#   >> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 apps/AY2latent_bci/train_compile_fwd.py config=apps/AY2latent_bci/configs/config_bci.yaml
#   >> CUDA_VISIBLE_DEVICES=1,5,6,7 torchrun --nproc_per_node=4 apps/AY2latent_bci/train_compile_fwd.py config=apps/AY2latent_bci/configs/config_bci.yaml

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
# from torch.distributed.tensor import distribute_tensor, Shard, Replicate
import torch._dynamo

import numpy as np
from scipy.fft import rfft, rfftfreq

# from lingua.apps.AY2latent.data_lean import STFTProcessor
from lingua.args import dataclass_from_dict, dump_config, flatten_dict
from lingua.checkpoint import CheckpointArgs, CheckpointManager, load_from_checkpoint

from apps.AY2latent_bci.eeg_data import (
    EEGProcessor, 
    BCIDatasetArgs, 
    create_dataloader_v2
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


# (CW) - to deal with compile graph break in model.sample in encoder.forward with mask from create_bidi_mask # MIGHT HAVE FIXED IT ??
torch._dynamo.config.capture_scalar_outputs = True  # (OR) env TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1



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
    # eval: Optional[Any] = None                                    # (CW) - uncomment if you dont want to run evals.
    eval: BCIDatasetArgs = field(default_factory=BCIDatasetArgs)    # (CW) - comment out if you dont want to run evals?

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

    track_dataset_stats: bool = False # (CW) to track dataset statistics like mean, std, max, min, median, 95CI of the batch in training loop.
    

@torch.compile()
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
    # for data_path in args.data.data_paths:
    #     assert os.path.exists(data_path), f"{data_path} doesn't exist" # (CW) - replaced with this second (OLD)
    #
    # assert os.path.exists(args.data.data_path), f"{args.data.data_path} doesn't exist" # (CW) - was this first


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

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    ## (CW) - make sure duplicated args in config are set to same value. NEED MORE!
    assert(
        args.model.dont_noise_chan_xyz ==
        args.data.dont_noise_chan_xyz ==
        args.eval.dont_noise_chan_xyz
    )
    assert(
        args.model.num_fine_time_pts ==
        args.data.num_fine_time_pts ==
        args.eval.num_fine_time_pts
    )
    assert(
        args.model.stft_global_sigma ==
        args.data.stft_global_sigma ==
        args.eval.stft_global_sigma
    )
    assert(
        args.data.masked_in_decoder ==
        args.eval.masked_in_decoder
    )

    # print(f"Inside validate_train_args, do asserts about config")
    # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)


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

    eeg_signal = eeg_signal.permute(0,2,1)
    if polyphase_factor > 1:
        print("Need to implement polyphase.")
    return eeg_signal


# def ChannelNorm(batch, verbose=False):
#     """
#     NOT USING ANYMORE. WENT BACK AND CLEANED UP THE DATA.
#     Normalizing each channel in each sample across time.
#  
#     (1). subtract mean
#     (2). divide by std
#     (3). clip vals > 3*sig
#     """
#
#     mu = batch.mean(dim=2).unsqueeze(-1)
#     sig = batch.std(dim=2).unsqueeze(-1)
#     batch_normed = ( batch - mu ) / (sig + 1e-8) 
#
#     # Clip things above 3sigma
#     outliers = abs(batch_normed)>3.0
#     batch_normed = torch.clamp(batch_normed, min=-3.0, max=3.0)
#
#     if verbose:
#         mu2 = batch_normed.mean(dim=2).unsqueeze(-1)
#         sig2 = batch_normed.std(dim=2).unsqueeze(-1)
#         print(f"ChannelNorm: (Bef --> Aft): average (mu, sig) = ({abs(mu).mean().item():0.2f}, {abs(sig).mean().item():0.2f})", end="")
#         print(f" --> ({abs(mu2).mean().item():0.2f}, {abs(sig2).mean().item():0.2f})", end=" ")
#         print(f"Data > 3sig = {100*outliers.sum().item()/batch_normed.numel():0.4f}%")
#
#     return batch_normed, mu, sig




def train(args: TrainArgs):
    with ExitStack() as context_stack:
        # tokenizer = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
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

        # print(f"In setup_torch_distributed")
        # print(f"{args.distributed=}")
        setup_torch_distributed(args.distributed)
        # print(f"Finish setup_torch_distributed")

        world_mesh = get_device_mesh(args.distributed)
        logger.info(f"Starting job: {args.name}")


        # print(f"After get_device_mesh, world_mesh: {world_mesh}")
        # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

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
        with torch.device("meta"):
            model = EncoderDecoder(args.model)
        logger.info("Model is built !")

        # print("In train - what is dp_degree?", dp_degree)
        # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)
        

        model_param_count = get_num_params(model)

        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=[],#build_fsdp_grouping_plan(args.model),
            # fsdp_grouping_plan=build_fsdp_grouping_plan(args.model),
            # tp_parallelize=tp_parallelize,
            no_recompute_ops=None,#get_no_recompute_ops(),
        )

        # Once we shard the model on different gpus we can actually initialize the model
        # First we create empty tensors of the correct shapes
        model = model.to_empty(device=f"cuda:{torch.cuda.current_device()}")
        # Then we init the model. Please make sure this function initializes *ALL* parameters
        # and buffers, otherwise you will have random values in the unitialized tensors
        # which will silently fail (give nan gradients for example)

        if False: # (CW) - comment out checkpoint loading for now.  was - if args.checkpoint.init_ckpt_path:
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
        if args.checkpoint.init_ckpt_path:
            _ = checkpoint.load(model, optimizer, train_state, world_mesh, Path(args.checkpoint.init_ckpt_path))
        else:
            _ = checkpoint.load(model, optimizer, train_state, world_mesh)


        ## (CW) - REBUILD OPTIMIZER AND TRAIN STATE AFTER CHECKPOINT LOAD if not loading optimizer state
        if not args.checkpoint.load_optimizer_state: # True:
            # build optimizer after apply parallelisms to the model
            optimizer, scheduler = build_optimizer(model, args.optim, args.steps,)
            # data_loader_state = init_dataloader_state_from_args(
            #     args.data, dp_rank, dp_degree
            # )

            train_state = TrainState(
                step=143000, # (CW) - set to last step of checkpoint HARDCODED FOR NOW.
                acc_step=0,
                # data_loader_state=data_loader_state,
                scheduler=scheduler,
            )

            train_state.scheduler.last_epoch=143000 #train_state.step (CW) - HARDCODED FOR NOW.



        # ## (CW) - Do This Here?:
        # check_model_value_range(model, range=10.0, std=1.0)


        # print(f" OUTSIDE IN TRAIN FUNCTION, AFTER CHECKPOINT LOAD")
        # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)


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
        # (CW) - NOTE: Standardized these with:
        #   1. create_dataloader in worker_init_fn and
        #   2. EEGDataset_v2.__iter__
        rank_seed = int(args.seed + (1e3 * dp_rank))
        torch.manual_seed(rank_seed)
        torch.cuda.manual_seed(rank_seed) 
        np.random.seed(rank_seed) # Also make numpy and random seeds unique per rank
        random.seed(rank_seed)

        logger.info(f"Setting torch seed to {rank_seed} for rank {dp_rank}")
        
        # train loop
        model.train()
        metric_logger = context_stack.enter_context(
            MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args)
        )
        # data_loader = context_stack.enter_context(
        #     create_dataloader(
        #         args.data,
        #     )
        # )


        # args.data.load_csv() (CW) - old audio stuff.
        print("Entering create dataloader on rank", dp_rank)
        data_loader = create_dataloader_v2(args.data, args.seed, dp_rank)
        print("Finishing create dataloader on rank", dp_rank)

        ## (CW) - debugging
        if False:

            # ## OLD WAY.
            # data_loader_eval = create_dataloader(args.eval, args.seed, dp_rank)
            # #
            # for idx,batch in enumerate(data_loader):
            #     print(f"\tTrain data loader {dp_rank=} : {idx=}: {data_loader=} : {batch.keys()=} : {batch['eeg_signal'][0,0,0]}")
            #     if idx>5:
            #         break
            # #
            # for idx,batch in enumerate(data_loader_eval):
            #     print(f"\tEval  data loader {dp_rank=} : {idx=}: {data_loader_eval=} : {batch.shape} : {batch[0,0,0]}")
            #     if idx>5:
            #         break
            # #
            # print("After creating dataloader and batch iterator for eval")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)


            ## NEW WAY.
            data_loader_eval = create_dataloader_v2(args.eval, args.seed, dp_rank)
            #
            print(f"\tTrain data loader {dp_rank=}: {data_loader=}") #: {batch.keys()=}")
            for idx,batch in enumerate(data_loader):
                print(f"\t\t {idx=}: eeg_signal: {type(batch['eeg_signal'])}, {batch['eeg_signal'].shape}")
                print(f"\t\t {idx=}: chan_pos:   {type(batch['chan_pos'])}, {batch['chan_pos'].shape}")
                print(f"\t\t {idx=}: ids:        {type(batch['ids'])}, {batch['ids'].shape}")
                print(f"\t\t {idx=}: dataset_id: {type(batch['dataset_id'])}, {batch['dataset_id'].shape}")
                print(f"\t\t----")
                if idx>=3:
                    break
            #
            print(f"\tEval data loader {dp_rank=}: {data_loader_eval=}") #: {batch.keys()=}")
            for idx,batch in enumerate(data_loader_eval):
                print(f"\t\t {idx=}: eeg_signal: {type(batch['eeg_signal'])}, {batch['eeg_signal'].shape}")
                print(f"\t\t {idx=}: chan_pos:   {type(batch['chan_pos'])}, {batch['chan_pos'].shape}")
                print(f"\t\t {idx=}: ids:        {type(batch['ids'])}, {batch['ids'].shape}")
                print(f"\t\t {idx=}: dataset_id: {type(batch['dataset_id'])}, {batch['dataset_id'].shape}")
                print(f"\t\t----")
                if idx>=3:
                    break
            #
            print("After creating dataloaders and batch iterators")
            import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)


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


        # batch_iterator = make_batch_iterator_old(data_loader) # (CW) - old way with Dataset
        batch_iterator = make_batch_iterator(data_loader) # (CW) - replacing with this for IterableDataset
        print("Entering create batch iterator on rank", dp_rank)


        # (CW) - if you want to run evals, create the dataloader and batch iterator here.
        if args.eval is not None:
            data_loader_eval = create_dataloader_v2(args.eval, args.seed, dp_rank) #create_dataloader(args.eval, args.seed+epoch+1+dp_rank, dp_rank)
            batch_iterator_eval = make_batch_iterator(data_loader_eval)

            print("After creating dataloader and batch iterator for eval")
            print(f"{dp_rank=}")
            print(f"{data_loader=}")
            print(f"{batch_iterator=}")
            print(f"{data_loader_eval=}")
            print(f"{batch_iterator_eval=}")
            # batch_eval = next(batch_iterator_eval)
            # print(f"{batch_eval.keys()=}")
            # for idx,batch in enumerate(data_loader_eval):
            #     print(f"{dp_rank=} : Inside data_loader_eval {idx}, {batch.keys()=}, {batch['eeg_signal'].shape=}") #, {batch['dataset_id']=}, {batch['ids']=}")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)


        torch_profiler = None #context_stack.enter_context(
        #     maybe_run_profiler(args.dump_dir, model, args.profiling)
        # )

        #make sure all model parameters require gradients
        for p in model.parameters():
            p.requires_grad = True

        # model.encoder.init_generator(torch.cuda.current_device())

        data_processor = EEGProcessor(args.data).to(torch.cuda.current_device())

        loss_weights = None

        loss_weights = {"mmd": args.encoder_mmd_weight, "decoder_rf_loss": args.decoder_loss_weight}

        #wrap all loss_weights into torch.tensor
        loss_weights = {k: torch.tensor(v, requires_grad=False).cuda(non_blocking=True) for k, v in loss_weights.items()}

        #partial process_batch_data with the stuff we have
        process_batch_data_compiled = functools.partial(
            process_batch_data,
            data_processor=data_processor,
            loss_weights=loss_weights,
        )
        
        process_batch_data_compiled = torch.compile(process_batch_data_compiled,)# mode="reduce-overhead", fullgraph=True)

        # Compile optimizer step
        # @torch.compile(fullgraph=False,)# dynamic=True)
        # @torch.compiler.set_stance("force_eager")
        def optimizer_do_step():
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()


        def convert_dtensor_model_to_tensor_model(dtensor_model: torch.nn.Module, args, device) -> torch.nn.Module:
            """
            Converts a model whose parameters are DTensors back to a model with regular torch.Tensor parameters.
            And puts model on same device as input data is.
            """
            regular_model = type(dtensor_model)(args)
            dtensor_state_dict = dtensor_model.state_dict()
            regular_state_dict = {}
            for key, value in dtensor_state_dict.items():
                if isinstance(value, torch.distributed.tensor.DTensor):
                    regular_state_dict[key] = value.to_local() # Convert DTensor to its local torch.Tensor representation
                else:
                    regular_state_dict[key] = value # If it's already a regular tensor, keep it.
            regular_model.load_state_dict(regular_state_dict)
            regular_model.to(device)

            # print("Inside convert_dtensor_model_to_tensor_model")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

            return regular_model

    
        def fwd_eval(batch, model_inference):
            """
            Run MSE evaluation harness on the batch of training data.
            Returns MSE_eeg, MSE_fft, MSE_latents.
            """  

            # print(f"Inside fwd_eval. At very beginning")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)
   
            
            with torch.no_grad(): 
                batch = data_processor.process(**batch)                             #  > option 3. (CW)
            
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()} 

            # # (CW) - swap the channel and time dimensions [B,T,C] -> [B,C,T] so that tok_embeddings linear layer does not mix channels.
            # if args.model.DONT_MIX_CHANNELS:
            #     batch['encoder_input'] = batch['encoder_input'].transpose(1, 2)[:,:,:args.model.num_fine_time_pts]  # and take only first 0.5 seconds (128 samples at 256Hz)
            #     batch['decoder_input'] = batch['decoder_input'].transpose(1, 2)[:,:,:args.model.num_fine_time_pts]
            #     if batch['target'] is not None:
            #         batch['target'] = batch['target'].transpose(1, 2)[:,:,:args.model.num_fine_time_pts]


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
                print(f"Dont understand {self.tok_idx_type=} and {self.rope_dim}")
                die

            # print(f"{args.model.tok_idx_type=} and {args.model.rope_dim=}")


            # print(f"Inside fwd_eval,")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

            with torch.no_grad():
                z, _ = model_inference.sample(
                                encoder_input=batch['encoder_input'].unsqueeze(0),
                                seq_lens=batch['seq_lens'],
                                tok_idx=tok_idx,
                )    


            # (1). For encoder consistency MSE, push encoder input and z back through model.encoder 
            # z_masked = z*batch['freq_masks']
            z_masked = z
            with torch.no_grad():
                # latent_data, _  = model_inference.encoder(batch['encoder_input']) # (CW) - was this
                latent_data, _ = model_inference.encoder(
                                        token_values=batch['encoder_input'].unsqueeze(0), 
                                        seq_lens=batch['seq_lens'],
                                        tok_idx=tok_idx,
                )
                #
                # latent_recon, _ = model_inference.encoder(z_masked) # (CW) - was this
                latent_recon, _ = model_inference.encoder(
                                        token_values=z_masked, 
                                        seq_lens=batch['seq_lens'],
                                        tok_idx=tok_idx,
                )

            #
            # push data to CPU
            latent_data = latent_data.cpu().numpy() 
            latent_recon = latent_recon.cpu().numpy() 
            model_output = z.cpu().float().numpy() 
            model_input = batch['encoder_input'].cpu().numpy()   # signal with dropout applied
            eeg_signal = batch['eeg_signal'].cpu().numpy()       # original signal without dropout applied


            # print("Inside fwd_eval, after model.encoder twice")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

            try:
                bad_mask = batch['freq_masks'].cpu().numpy()
            except:
                bad_mask = np.ones((1, batch['seq_lens'].sum().item(), 1), dtype=bool) # [B=1, C=seqlen, 1]


            # Do everything in this chunk. Not breaking apart into samples again.
            # Probably want to reintroduce sample by sample eval again. Esp to plot samples.
            MSE_latent = [np.abs(latent_data - latent_recon).mean()]
            MSE_EEG = [np.abs(eeg_signal - model_output.squeeze(0)).mean()] # (CW) - USE NON-DROPPED-OUT DATA. was this: [np.abs(model_input - model_output.squeeze(0)).mean()]
            #
            STD_latent_data = [latent_data.std()]
            STD_latent_recon = [latent_recon.std()]
            STD_EEG_data = [eeg_signal.std()]
            STD_EEG_recon = [model_output.squeeze(0).std()]
            #
            MSE_FFT = [-1]
            bad_chans = [-1]




            # OLD WAY WE WERE DOING THINGS. WILL HAVE TO REDO AFTER PACKING NOW.
            if False:
                
                # Compute MSE between latent_data and latent_recon (ON CPU!)
                MSE_samp_latent = []
                samp_bad_chans = []
                for samp in range(args.eval.batch_size):
                    latent_data_sample = latent_data[samp]
                    latent_recon_sample = latent_recon[samp]
                    MSE = np.abs(latent_data_sample - latent_recon_sample).mean() # mean square error btwn data and reconst
                    MSE_samp_latent.append(MSE.item())
                    samp_bad_chans.append( (bad_mask[samp]==False).sum().item() )

                #
                # Compute FFT of raw EEG signal (model_output and model_input) (on CPU!)
                # NOTE: Only compute FFT if we are mixing channels in tok_embeddings and applying attention across time points.
                if args.model.DONT_MIX_CHANNELS==False:
                    fs = args.eval.sample_rate
                    num_t = model_input.shape[-1] # (CW): Is this right? - was this: args.eval.seq_len
                    freqs = rfftfreq(num_t, 1/fs)
                    #
                    fft_data = np.abs(rfft(model_input, axis=1))
                    data_norms = np.linalg.norm(fft_data, axis=1) 
                    fft_data = fft_data / (data_norms[:, np.newaxis, :] + 1e-6)
                    #
                    fft_reconst = np.abs(rfft(model_output, axis=1))
                    reconst_norms = np.linalg.norm(fft_reconst, axis=1) 
                    fft_reconst = fft_reconst / (reconst_norms[:, np.newaxis, :] + 1e-6)
                #
                # Compute MSE between model_output and model_input EEGs & FFTs (ON CPU!)
                MSE_samp_EEG = []
                MSE_samp_FFT = []
                for samp in range(args.eval.batch_size):

                    # (1). Compute MSE between model_input & model_output for EEG for single samples
                    model_in = model_input[samp]
                    model_out = model_output[samp]
                    #
                    # Mask out bad channels in EEG (ones that are all-zero in data).
                    mask_eeg = np.broadcast_to(bad_mask[samp], model_in.shape)
                    good_data_eeg = model_in[mask_eeg]
                    good_recon_eeg = model_out[mask_eeg]
                    #
                    MSE = np.abs(good_data_eeg - good_recon_eeg).mean() # mean square error btwn data and reconst

                    # (2). Compute MSE between model_input & model_output for FFT for single samples
                    if args.model.DONT_MIX_CHANNELS==False:
                        fft_sample_data = fft_data[samp]
                        fft_sample_recon = fft_reconst[samp]
                        #
                        # Mask out bad channels in FFT (ones that are all-zero in data).
                        mask_fft = np.broadcast_to(bad_mask[samp], fft_sample_data.shape)
                        good_data_fft = fft_sample_data[mask_fft]
                        good_recon_fft = fft_sample_recon[mask_fft]
                        #
                        MSEf = np.abs(good_data_fft - good_recon_fft).mean() # mean square error btwn data and reconst FFTs
                    else:
                        MSEf = torch.tensor(-1.0)

                    # print(f"MSE (FFTdata - FFTreconst) = {MSEf:0.2f}")

                    MSE_samp_EEG.append(MSE.item())
                    MSE_samp_FFT.append(MSEf.item())    

            # print("Inside fwd_eval, at the very end!!")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

            # Delete tensors off GPU device
            del latent_data, latent_recon, bad_mask, model_output, model_input, z, batch            

            return MSE_EEG, MSE_FFT, MSE_latent, STD_latent_data, STD_latent_recon, STD_EEG_data, STD_EEG_recon, bad_chans



        
        @torch.compile() # THIS WORKS WITH AND WITHOUT COMPILE COMMENTED OUT. IT SEEMS. (CW)
        def fwd_step(batch):

            # print(f"Inside fwd_step.")
            # print(f"{batch.keys()=} and {batch['eeg_signal'].shape=}")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)


            # batch, loss_weights_batch = process_batch_data_compiled(batch, train_state.step) # (CW) - was this. REINSTATE THE COMPILED VERSION.
            #
            # batch, loss_weights_batch = process_batch_data(batch, train_state.step) # - option 2 (CW) - maybe change to noncompiled version ???
            #
            with torch.no_grad():                                                   # \
                batch = data_processor.process(**batch)                             #  > option 3. (CW)
            loss_weights_batch = loss_weights                                       # /

            eeg_signal = batch.pop('eeg_signal', None)          # Not used for training. Used for eval.

            # print("After reshaping eeg_signal and EEGProcessor, batch is:")
            # for k,v in batch.items():
            #     print(f"\t{k} - {v.shape}")
            # print(" ")
            # print(f"{type(model)=}")
            # import IPython; print('\n\n Debug:'); IPython.embed(); import time;  time.sleep(0.3)

            # if torch.isnan(batch['encoder_input']).any():
            #     logger.error("NaNs in encoder_stft")
            # if torch.isnan(batch['decoder_input']).any():
            #     logger.error("NaNs in decoder_input_stft")
            # if torch.isnan(batch['target']).any():
            #     logger.error("NaNs in target")
            # if torch.isnan(batch['t']).any():
            #     logger.error("NaNs in t")
            # if torch.isnan(batch['time_masks']).any():
            #     logger.error("NaNs in time_masks")
            # if torch.isnan(batch['freq_mask']).any():
            #     logger.error("NaNs in freqmask") 


            ## (CW). MOVED THIS HERE. IT WAS BEFORE FWD_STEP WHEN KEYS IN BATCH ARE JUST 'EEG_SIGNAL'. SHOULD IT HAPPEN HERE?
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()} # Q: SHOULD THIS HAPPEN INSIDE FWD_STEP, AFTER EEGPROCESSOR.process YEILDS encoder_input, decoder_input, target, t?





            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

            # # MIMIC WHAT HAPPENS IN MODEL

            # encoder_input = batch['encoder_input']
            # decoder_input = batch['decoder_input']
            # target = batch['target']
            # t = batch['t']
            # chan_pos = batch['chan_pos']
            # chan_pos_discrete = batch['chan_pos_discrete']
            # chan_id = batch['chan_id']
            # seq_lens = batch['seq_lens']
            # t_coarse = batch['t_coarse']

            # # Reintroduce the batch dimension. Come back to: I forget, Do I actually need to do this in the first place? Is it needed for something in Encoder?
            # if encoder_input.ndim==2:
            #     encoder_input = encoder_input.unsqueeze(0)
            #     target = target.unsqueeze(0) # doing to get rid of broadcast warning from DecoderTransformer.compute_losses
            #     chan_pos = chan_pos.unsqueeze(0)
            #     chan_pos_discrete = chan_pos_discrete.unsqueeze(0)
            #     chan_id = chan_id.unsqueeze(0)
            #     t_coarse = t_coarse.unsqueeze(0) # We are just undoing this later...
            #     # seq_ids = seq_ids.unsqueeze(0)



            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


            # print(f"In train_fwd_compile.py, in fwd_step, before model forward.")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)



            # print(f"{batch['encoder_input'].shape=}")
            _, enc_loss, dec_loss = model(**batch)

            # print(f"In train_fwd_compile.py, in fwd_step, after model forward.")
            # print(f"{loss_weights_batch=}")
            # print(f"{enc_loss=}")
            # print(f"{dec_loss=}")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)


            if args.grad_acc_steps > 1:
                model.set_requires_gradient_sync(train_state.acc_step == 0)

            loss = sum(v * loss_weights_batch[k] for k, v in enc_loss.items()) + sum(v * loss_weights_batch[k] for k, v in dec_loss.items())
            loss = loss / args.grad_acc_steps
            loss.backward()

            grad_norm = -1.0
            if train_state.acc_step == 0:
                
                # with torch.compiler.set_stance("force_eager"):
                if True: 
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.optim.clip, foreach=True)
                    # print(f"DOING GRADIENT NORM CLIPPING.") # --> Original gradient norm: {grad_norm:.4f}")

                optimizer_do_step()
                # optimizer.step()
                # optimizer.zero_grad()
                train_state.step += 1
            return loss, grad_norm, enc_loss, dec_loss
        
        nwords_since_last_log = 0
        time_last_log = timer()
        gc.collect()
        #clear cuda cache
        # torch.cuda.empty_cache()
        if hasattr(optimizer, "train"):
            optimizer.train()

        first_print = True
        fwd_compiled = False

        while train_state.step < args.steps:

            # print(f"INSIDE TRAINING LOOP, WITH {train_state.step} STEPS TAKEN")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

            # We constrain train_state.acc_step to be in range 0 to args.grad_acc_steps - 1
            train_state.acc_step += 1
            train_state.acc_step = train_state.acc_step % args.grad_acc_steps

            # get training data batch
            curr_lr = float(optimizer.param_groups[0]["lr"])
            data_load_start = timer()
            #
            try:
                # dbs = timer()
                batch = next(batch_iterator)
                # dbt = round(timer() - dbs, 4)
                # print(f"Getting next batch took {dbt} secs!")
            except Exception as e:
                print("The dataloader passed away 👻!", e)
                # data_loader = create_dataloader_v2(args.data, args.seed+epoch+1+dp_rank, dp_rank) # getting rid of nonlocal epoch tracking
                data_loader = create_dataloader_v2(args.data, args.seed+train_state.step+1+dp_rank, dp_rank) # REVISIT THIS!!! SEEDS.
                batch_iterator = make_batch_iterator(data_loader)
                batch = next(batch_iterator)

            # print(f"In train step, what is batch?")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)            

            # Doing channel masking based on freq_mask input into EncoderDecoder.forward in transformer.py 
            mask_bad_chans = False # (CW) - There shouldnt be any more zero'ed out channels. 
            if mask_bad_chans:
                # print(f"Masking out any bad (all zero) channels.")
                bad_mask = batch['eeg_signal'].abs().sum(axis=1)!=0 # mask out channels that are ALL ZERO --> freq_masks [B,1,C] that is [0 if bad, 1 if good]

                print(f"In mask_bad_chans")
                import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

                # (CW) - swap the channel and time dimensions [B,T,C] -> [B,C,T] so that tok_embeddings linear layer does not mix channels.
                batch['freq_masks'] = bad_mask.unsqueeze(2).int() # [B,C,1]
                pct_unmasked = 100*(batch['freq_masks'].sum().item()/batch['freq_masks'].numel()) # percent of batch that is masked due to bad or missing channels.
            else:
                # batch['freq_masks']=None
                pct_unmasked = 101

            # NOTE: pop takes these out of batch. (CW) - if left in, breaks things below and not training on these.
            batch_idx = batch.pop('idx', None)                  
            batch_dataset_id = batch.pop('dataset_id', None)

            # This is sorta dropout rate, but mixes two effects (samples with no dropout - p=0.33 - and num-chans dropped in DO samples - rand uniform). 
            pct_dropout = batch['chan_dropout'].sum().item()/batch['chan_dropout'].numel() 

            # (CW) - Compute some statistics on Data Batch to save into logger and plot out in WandB
            if args.track_dataset_stats:
                # apply the "bad_chan_mask" to eeg_signal before computing stats so you only include good chans in the stat.
                mask = bad_mask.unsqueeze(1).expand_as(batch['eeg_signal'])
                good_signal = batch['eeg_signal'][mask]
                #
                if good_signal.numel() > 0:
                    batch_mean = good_signal.mean().item()
                    batch_std = good_signal.std().item()
                    batch_max = good_signal.max().item()
                    batch_min = good_signal.min().item()
                    #
                    percentiles = torch.quantile(abs(good_signal.float()), torch.tensor([0.5, 0.95]))
                    batch_median = percentiles[0].item()
                    batch_95CI = percentiles[1].item()
                else:
                    batch_mean = 0 # if there are no good channels in whole batch.
                    batch_std = 0
                    batch_max = 0
                    batch_min = 0
                    batch_median = 0
                    batch_95CI = 0
                #
                # batch_ids = batch.pop('ids', None)                      # NOTE: pop takes them out of batch. (CW) - if left in, breaks things below and not training on these.
                # batch_chan_mu = batch.pop('chan_mu', None)              # Not recording in WandB yet. 
                # batch_chan_sig = batch.pop('chan_sig', None)            # Not recording in WandB yet. Note: These dont exist if not doing ChanNorm, but None catches non-existence.
                
                # extract mode of dataset id from all samples. 
                ds_values, ds_counts = torch.unique(batch_dataset_id, return_counts=True)
                mode_idx = torch.argmax(ds_counts)
                dataset_id_mode = ds_values[mode_idx].item()
                dataset_id_counts = len(ds_values)

            if first_print:
                logger.info(f"Batch keys: {batch.keys()}")
                logger.info(f"Batch shapes: {[(k, v.shape) for k, v in batch.items()]}") # NOT WORKING WITH SEQ_BEGS list.
                first_print = False

            # batch = process_batch_data(args, train_state, data_processor, loss_weights, channel_loss_weighting, distill_model, resampler, batch)
            # batch, loss_weights_batch = process_batch_data_compiled(batch, train_state.step)

            data_load_time = round(timer() - data_load_start, 4)



            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                logger.info("garbage collection")
                # we do garbage collection manually otherwise different processes
                # run the GC at different times so they slow down the whole pipeline
                gc.collect()

            # encoder_stft = batch['encoder_stft']
            # decoder_input_stft = batch['decoder_input_stft']
            # target = batch['target']
            # t = batch['t']

            #check for NaNs in the input

            # bsz, _, seqlen, c = decoder_input_stft.shape

            # forward
            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            # This is an automatic probe that will compute statistics
            # of all linears' inputs, weights and outputs
            # along with attention logits and entropy
            # both in forward and backward pass
            # if (args.probe_freq is not None) and every_n_steps(
            #     train_state, args.probe_freq, acc_step=1 % args.grad_acc_steps
            # ):
            #     # Here we do a fake forward and backward pass on a smaller
            #     # batch size to avoid OOM
            #     # This assumes the model has no stateful layers (batch norm..)
            #     assert (
            #         next(model.parameters()).grad is None
            #     ), "Can't probe model if grads are not reset"

            #     with probe:
            #         probe.metadata = {
            #             "it": train_state.step,
            #             "global_step": train_state.step,
            #             "loop": "lingua",
            #         }
            #         # Non compiled model uses roughly 2x memory in our exps
            #         # So we divide bsz by 2 or seqlen by 2
            #         probe_bsz = max(1, bsz // 2)
            #         probe_seq = seqlen if (bsz // 2 >= 1) else (seqlen // 2)
            #         probe_loss = model(
            #             input_ids[:probe_bsz, :probe_seq],
            #             labels[:probe_bsz, :probe_seq],
            #         )
            #         probe_loss.backward()
            #         # We zero grads to cancel this fake step
            #         optimizer.zero_grad()

            #     assert (
            #         next(model.parameters()).grad is None
            #     ), "Probe model shouldn't have grads at this point"

            # _, enc_loss, dec_loss = model(**batch)
            # print("Encoder shit", enc_loss)
            # print("Decoder shit", dec_loss)
            # enc_loss and dec_loss are both dicts then do weighted sum by matching keys against loss_weights
            # loss = sum(v * loss_weights_batch[k] for k, v in enc_loss.items()) + sum(v * loss_weights_batch[k] for k, v in dec_loss.items())

            # if args.grad_acc_steps > 1:
            #     model.set_requires_gradient_sync(train_state.acc_step == 0)

            # # We scale loss with grad_acc_steps so the gradient is the same
            # # regardless of grad_acc_steps
            # loss = loss / args.grad_acc_steps

            # # backward on scaled loss to create scaled gradients
            # loss.backward()
            # # For logging we undo that scaling
            # loss = loss.detach() * args.grad_acc_steps

            # # optimizer step
            # grad_norm = -1.0
            # if train_state.acc_step == 0:
            #     grad_norm = torch.nn.utils.clip_grad_norm_(
            #         model.parameters(), max_norm=args.optim.clip, foreach=True
            #     )

            #     grad_norm = (
            #         grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm
            #     ).item()

            #     optimizer_do_step()
            #     # optimizer.step()
            #     scheduler.step()
            #     # optimizer.zero_grad()
            #     train_state.step += 1

            # updates the scale for next iteration
            # training iteration complete
            loss, grad_norm, enc_loss, dec_loss = fwd_step(batch)
            end_timer.record()

            torch.cuda.synchronize()
            # if not fwd_compiled:
            #     # fwd_step = torch.compile(fwd_step)
            #     #mark all grads with mark_static_address
            #     for p in model.parameters():
            #         if p.grad is not None:
            #             mark_static_address(p.grad)
            #         else:
            #             logger.warning("Gradient is None")
            #     logger.info("Gradients are marked with static address")

            #     #do same to optimizer params
            #     for p in optimizer.param_groups[0]['params']:
            #         if p.grad is not None:
            #             mark_static_address(p.grad)
            #         else:
            #             logger.warning("Optimizer param is None")
            #     logger.info("Optimizer params are marked with static address")
            #     fwd_compiled = True

            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)
            loss = loss.item() * args.grad_acc_steps
            try:
                grad_norm = (
                        grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm
                ).item()
            except:
                pass


            # # if profiler is active
            # if torch_profiler:
            #     xformers.profiler.step()

            


                

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            #
            # (CW) - SAVE CHECKPOINT if you are at a checkpoint step or if you are at an eval step.
            #       Note: Not saving checkpoint on eval steps.
            saved = False
            if (
                every_n_steps(train_state, args.checkpoint.dump.every, acc_step=0) 
            ):
            #     or every_n_steps(train_state, args.checkpoint.eval.every, acc_step=0)
            # ): #(CW) Commented out - Dont save checkpoint on eval steps.: # 
            
                #check if optimizer has .eval() method
                if hasattr(optimizer, "eval"):
                    optimizer.eval()

                # if loss_weights['decoder_repa_loss'] != args.decoder_repa_weight:
                #     loss_weights['decoder_repa_loss'] = args.decoder_repa_weight
                #     logger.info("Setting decoder repa loss to {}".format(args.decoder_repa_weight))

                # original_csv_contents = args.data.csv_contents # (CW) - 1/2. this errors and seems useless.
                args.data.csv_contents = []

                saved = checkpoint.save(
                    model,
                    optimizer,
                    train_state,
                    args,
                    device_mesh=world_mesh,
                )
                # args.data.csv_contents = original_csv_contents # (CW) - 2/2. this errors and seems useless.


                # Save and reload the model, optimizer, and train state to verify that they are the same.
                if False:
                    
                    print("After loading checkpoint - verifying that model, optimizer, and train state are the same.")
                    import IPython; print('\n\nDebug: after loading checkpoint'); IPython.embed(); import time;  time.sleep(0.3)

                    loaded_state_dict = checkpoint.load(model, optimizer, train_state, world_mesh) # CW - Load the last saved checkpoint.


                    # (1). Check that the model state_dict is the same.
                    for key, value in model.state_dict().items():
                        if not (loaded_state_dict["model"][key] == value).all().item():
                            print(f"{key=}, {value.shape=}, {value.type=}")
                            print(f'{loaded_state_dict["model"][key].shape=}, {loaded_state_dict["model"][key].type=}')
                            print("--------------------------------")



                    # (2). Check that the optimizer states state_dict is the same.
                    for key, value in optimizer.state_dict()["state"].items():
                        if not (
                        (value["step"] == loaded_state_dict["optim"]["state"][key]["step"]).all().item() and
                        (value["exp_avg"] == loaded_state_dict["optim"]["state"][key]["exp_avg"]).all().item() and
                        (value["exp_avg_sq"] == loaded_state_dict["optim"]["state"][key]["exp_avg_sq"]).all().item()
                        ):
                            print("FUCK! THE OPTIMIZER STATES ARE NOT THE SAME!")


                    # (3). Check that the optimizer param groups state_dict is the same.
                    assert len(optimizer.state_dict()["param_groups"]) == len(loaded_state_dict["optim"]["param_groups"])
                    assert len(optimizer.state_dict()["param_groups"]) == 1 # CW - Only one param group for now.

                    for key, value in optimizer.state_dict()["param_groups"][0].items():
                        if not (value == loaded_state_dict["optim"]["param_groups"][0][key]):
                            print("FUCK! THE OPTIMIZER PARAM GROUPS ARE NOT THE SAME!")
                            print(f"{key=}, {value=}") # {value.shape=},
                            print(f'{loaded_state_dict["optim"]["param_groups"][0][key]=}')
                            print("--------------------------------")

                            

                    #assert train_state.state_dict() == loaded_state_dict["train_state"]

                    








                if hasattr(optimizer, "eval"):
                    optimizer.train()



            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            #
            # (CW) - EVAL HARNESS - If eval is set, then run eval every args.checkpoint.eval.every steps.
            eval_ran=False
            eval_time = 0.0
            if args.eval is not None and every_n_steps(train_state, args.checkpoint.eval.every, acc_step=0):
                eval_ran = True
                eval_start = timer()
                eval_step = 0
                print(f"IN EVAL HARNESS")

                if hasattr(optimizer, "eval"):
                    optimizer.eval()

                # (CW) TODO: How to keep from re-compiling the model every time we eval?
                # Setup and compile model for inference
                model_inference = convert_dtensor_model_to_tensor_model(
                    model, 
                    args=args.model, 
                    device=f"cuda:{torch.cuda.current_device()}"  # Use local device, not global dp_rank
                ) 
                model_inference.sample = torch.compile(model_inference.sample)                          
                model_inference.encoder = torch.compile(model_inference.encoder)

                # Debugging Eval Dataloader (Iterable vs Non-iterable dataset)
                if False:
                    print(f"IN EVAL HARNESS: {dp_rank=}") # , {model_inference=}")
                    #
                    ## What device are the different models on? Seems right. dp_rank on GPU:dp_rank
                    ## But, we make the eval_dataloader way before this. Is that the problem?
                    for key, value in model_inference.state_dict().items():
                        print(f"{key=}, {value.device=}")

                    print(f"{dp_rank=}, {type(model)=}")
                    print(f"{dp_rank=}, {type(model_inference)=}")

                    print("Inside Eval Harness after making model_inference")
                    import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

                MSE_batch_EEG = []
                MSE_batch_FFT = []
                MSE_batch_latent = []
                STD_batch_latent_data = []
                STD_batch_latent_recon = []
                STD_batch_EEG_data = []
                STD_batch_EEG_recon = [] 
                batch_bad_chans = []

                # batch_eval = 1 # just to enter the loop
                # while batch_eval is not None: #
                while eval_step < args.eval.num_batches:
                    eval_step += 1
                    print(f"{eval_step=}")

                    # (CW) - Make eval dataloader to specifically use different eval EEG dataset
                    try:
                        batch_eval = next(batch_iterator_eval)
                    except Exception as e:
                        print("The eval dataloader passed away 👻!", e)
                        # data_loader_eval = create_dataloader(args.eval, args.seed+epoch+1+dp_rank, dp_rank) # got rid of nonlocal epoch tracking
                        data_loader_eval = create_dataloader_v2(args.eval, args.seed+eval_step+1+dp_rank, dp_rank)
                        batch_iterator_eval = make_batch_iterator(data_loader_eval)
                        batch_eval = next(batch_iterator_eval)


                    # if batch_eval is None:
                    #     print(f"Eval dataloader is exhausted after {eval_step} batches.")
                    #     break


                    # print("Inside Eval Harness before model.sample()")
                    # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

                    # Doing channel masking based on freq_mask input into EncoderDecoder.forward in transformer.py 
                    mask_bad_chans = False 
                    if mask_bad_chans:
                        # print(f"Masking out any bad (all zero) channels.")
                        bad_mask = batch_eval['eeg_signal'].abs().sum(axis=1)!=0 # mask out channels that are ALL ZERO --> freq_masks [B,1,C] that is [0 if bad, 1 if good]

                        # (CW) - swap the channel and time dimensions [B,T,C] -> [B,C,T] so that tok_embeddings linear layer does not mix channels.
                        if args.model.DONT_MIX_CHANNELS:
                            batch_eval['freq_masks'] = bad_mask.unsqueeze(2).int() # [B,C,1]
                        else:
                            batch_eval['freq_masks'] = bad_mask.unsqueeze(1).int() # [B,1,C]  

                        pct_unmasked = 100*(batch_eval['freq_masks'].sum().item()/batch_eval['freq_masks'].numel()) # percent of batch that is masked due to bad or missing channels.
                    else:
                        # batch['freq_masks']=None
                        pct_unmasked = 101

                    batch_eval_idx = batch_eval.pop('idx', None)                      # NOTE: pop takes them out of batch. (CW) - if left in, breaks things below and not training on these.
                    batch_eval_dataset_id = batch_eval.pop('dataset_id', None)        # NOTE: pop takes them out of batch. (CW) - if left in, breaks things below and not training on these.


                    # MSE_samp_EEG, MSE_samp_FFT, MSE_samp_latent, samp_bad_chans = fwd_eval(batch, model_inference)       # run MSE eval on training data batch
                    MSE_samp_EEG, MSE_samp_FFT, MSE_samp_latent, \
                        STD_samp_latent_data, STD_samp_latent_recon, \
                        STD_samp_EEG_data, STD_samp_EEG_recon, \
                        samp_bad_chans = fwd_eval(batch_eval, model_inference)    # run MSE eval on eval data batch


                    MSE_batch_EEG.extend(MSE_samp_EEG)
                    MSE_batch_FFT.extend(MSE_samp_FFT)
                    MSE_batch_latent.extend(MSE_samp_latent)
                    STD_batch_latent_data.extend(STD_samp_latent_data)
                    STD_batch_latent_recon.extend(STD_samp_latent_recon)
                    STD_batch_EEG_data.extend(STD_samp_EEG_data)
                    STD_batch_EEG_recon.extend(STD_samp_EEG_recon)
                    batch_bad_chans.extend(samp_bad_chans)

                    ## (CW) - To verify that different ranks are actually getting different data. (Seems they are.)
                    # print(f"{dp_rank=}, Some Stats on batch_eval eeg_signal: ")
                    # print(f"\t mean: {batch_eval['eeg_signal'].mean().item()}")
                    # print(f"\t std: {batch_eval['eeg_signal'].std().item()}")
                    # print(f"\t min: {batch_eval['eeg_signal'].min().item()}")
                    # print(f"\t max: {batch_eval['eeg_signal'].max().item()}")

                    # print("Inside Eval Harness after model.sample()")
                    # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3) 

                    # (CW) - THIS IS THE EVAL HARNESS THAT CAME WITH LINGUA CODEBASE. WE WANT SOMETHING DIFFERENT.
                    if False:
                        from apps.main.eval import (
                            launch_eval,
                            EVAL_FOLDER_NAME,
                            EvalArgs,
                        )

                        eval_args = dataclass_from_dict(EvalArgs, args.eval)

                        eval_args.global_step = train_state.step
                        eval_args.ckpt_dir = str(checkpoint.existing_saves[-1])
                        eval_args.dump_dir = str(
                            os.path.join(
                                args.dump_dir,
                                "evals",
                                EVAL_FOLDER_NAME.format(train_state.step),
                            )
                        )
                        eval_args.metric_log_dir = args.dump_dir
                        if args.async_eval_gpus is None:
                            launch_eval(eval_args)
                        elif get_is_master():
                            if wandb.run is not None and args.logging.wandb is not None:
                                eval_args.wandb = deepcopy(args.logging.wandb)
                            assert args.async_eval_gpus > 0
                            logger.info(f"Launching evals on {args.async_eval_gpus} gpus")
                            with clean_env():
                                launch_job(
                                    StoolArgs(
                                        asdict(eval_args),
                                        script="apps.main.eval",
                                        copy_code=False,
                                        nodes=args.async_eval_gpus // 8,
                                        qos="lowest",
                                    )
                                )


                if hasattr(optimizer, "eval"):
                    optimizer.train()

                eval_time = round(timer() - eval_start, 4)

                # # clean up inference model from GPU memory
                del model_inference
                torch.cuda.empty_cache()
                gc.collect()


            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            #
            # log metrics
            if every_n_steps(
                train_state,
                args.logging.freq,
                acc_step=None if args.logging.acc_freq else 0,
                acc_freq=args.logging.acc_freq,
            ):
                time_delta = timer() - time_last_log
                wps = nwords_since_last_log / (time_delta * args.distributed.tp_size)

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                total_acc_steps = (
                    args.grad_acc_steps * train_state.step + train_state.acc_step
                )
                ## (CW) - Comment out because time_masks not defined in batch - is None.
                # tokens_per_gpu = (
                #     total_acc_steps * args.data.batch_size * batch['time_masks'].sum().item() #args.data.seq_len
                # )
                # total_tokens = dp_degree * tokens_per_gpu

                # This is an estimate and the correct values may change
                # if you change the architecture
                # Use xformer's analyze profile trace to get actual measurement
                FLOPS = (
                    get_num_flop_per_token(
                        model_param_count,
                        args.model.n_layers,
                        args.model.dim,
                        args.data.seq_len,
                    )
                    * wps
                )

                # convert tensor to number for pandas WANDB below 🙄
                if batch_idx is not None:
                    batch_idx = batch_idx.to(torch.float32).mean().item()



                # Basic metrics to track and plot on WANDB
                metrics_dict = {
                    "global_step": train_state.step,
                    "acc_step": train_state.acc_step,
                    "speed": {
                        "wps": wps,
                        "FLOPS": FLOPS,
                        "curr_iter_time": curr_iter_time,
                        "data_load_time": data_load_time,
                    },
                    "optim": {
                        "grad_norm": grad_norm,
                        "lr": curr_lr,
                        # "total_tokens": total_tokens, # (CW) - Comment out bc time_mask is None
                    },
                    "memory": gpu_mem_stats._asdict(),
                    "data_stats": {
                        "idx": batch_idx,                   # which batch
                        "pct_unmasked": pct_unmasked,       # percent of batch that is masked out due to bad/missing chans.
                        "pct_dropout": pct_dropout,         # percent of batch that is masked out due to channel dropout.
                    },
                }

                # More computationally expensive metrics about datasets that we can track, but that slow down training loop.
                if args.track_dataset_stats:
                    metrics_dict["data_stats"].update({
                        # "ids": batch_ids,                     # which dataset index
                        "dataset_id_mode": dataset_id_mode,     # which dataset id is mode
                        "num_datasets": dataset_id_counts,      # how many unique datasets
                        "mean": batch_mean,
                        "std": batch_std,
                        "max": batch_max,
                        "min": batch_min,
                        "median": batch_median,
                        "95CI": batch_95CI,
                    })


                # (CW) - Add MSE metrics to metrics_dict if they exist or evals were run.
                if eval_ran:
                    metrics_dict["evals"] = {
                        "MSE_EEG": np.mean(MSE_batch_EEG).item(), 
                        "MSE_FFT": np.mean(MSE_batch_FFT).item(), 
                        "MSE_latent": np.mean(MSE_batch_latent).item(), 
                        "STD_latent_data": np.mean(STD_batch_latent_data).item(),
                        "STD_latent_recon": np.mean(STD_batch_latent_recon).item(),
                        "STD_EEG_data": np.mean(STD_batch_EEG_data).item(),
                        "STD_EEG_recon": np.mean(STD_batch_EEG_recon).item(),
                        "masked_chans": np.mean(batch_bad_chans).item(),
                        "num_eval_samples": len(MSE_batch_EEG)
                    }

                metrics = flatten_dict(metrics_dict, sep="/",)

                to_sync = {}
                to_sync["loss/out"] = loss#.item()
                # to_sync["dec_loss/out"] = dec_loss.item()
                for k, v in dec_loss.items():
                    to_sync[f"dec_loss/{k}"] = v.mean().item()
                for k, v in enc_loss.items():
                    to_sync[f"enc_loss/{k}"] = v.mean().item()
                metrics.update(dist_mean_dict(to_sync))

                #add loss weights to metrics
                # metrics.update({'loss_weights/'+k: v for k, v in loss_weights.items()})

                # print("Look at metrics - some pandas json error")
                # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

                if get_is_master(): 
                    metric_logger.log(metrics)

                gpu_memory_monitor.reset_peak_stats()
                nwords_since_last_log = 0
                time_last_log = timer()

                # print(f"Before logger. Add seqlen:")
                # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

                logger_str = (
                    f"step: {train_state.step}"
                    f"  seqlen: {batch['seq_lens'].sum().item()}"
                    # f"  acc: {train_state.acc_step}"
                    f"  loss: {round(loss,4):>7}"
                    f"  dec_loss: {' '.join([f'{k}: {round(v.mean().item(),4):>7}' for k, v in dec_loss.items()])}" # log dec losses separately
                    f"  enc_loss: {' '.join([f'{k}: {round(v.mean().item(),4):>7}' for k, v in enc_loss.items()])}"
                    f"  grad: {grad_norm:.2e}"
                    # f"  flops: {FLOPS:.2e}"
                    # f"  wps: {wps:.2e}"
                    f"  iter t: {curr_iter_time:>7}"
                    f"  data t: {data_load_time:>5}"
                    f"  eval t: {eval_time:>5}"  
                    f"  lr: {curr_lr:.2e}"
                    f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                    f"  pow: {gpu_mem_stats.power_draw/1000} W"
                    f"  batch-unmasked: {pct_unmasked:3.0f}%"
                )
                if args.track_dataset_stats:
                    logger_str += (
                        f"  dataset-ids: {ds_values.tolist()}"
                    )
                logger.info(logger_str)


            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            #
            # (CW) - Checkpoint if preemption is detected.
            if preemption_flag["flag"]:
                if not saved:
                    if hasattr(optimizer, "eval"):
                        optimizer.eval()
                    # original_csv_contents = args.data.csv_contents # (CW) - 1/2. this errors and seems useless.
                    args.data.csv_contents = []
                    checkpoint.save(
                        model,
                        optimizer,
                        train_state,
                        args,
                        device_mesh=world_mesh,
                    )
                    # args.data.csv_contents = original_csv_contents # (CW) - 2/2. this errors and seems useless.
                    if hasattr(optimizer, "eval"):
                        optimizer.train()
                requeue_slurm_job()
                sys.exit(0)


    # (CW) - Save checkpoint at the end of training if not saved yet.
    if not saved:
        if hasattr(optimizer, "eval"):
            optimizer.eval()
        # original_csv_contents = args.data.csv_contents # (CW) - this errors and seems useless.
        args.data.csv_contents = []
        checkpoint.save(
            model,
            optimizer,
            train_state,
            args,
            device_mesh=world_mesh,
        )
        if hasattr(optimizer, "eval"):
            optimizer.train()
    gc.collect()

# def process_batch_data(args, train_state, data_processor, loss_weights, channel_loss_weighting, distill_model, resampler, batch):
# @torch.compile()


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


    # print(f"I am in main before imports and before loading config.")
    # die

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

    train(cfg)


if __name__ == "__main__":
    main()
