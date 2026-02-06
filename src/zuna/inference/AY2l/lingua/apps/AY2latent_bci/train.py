# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

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

from apps.AY2latent.data_lean import(
    AudioDatasetArgs,
    create_dataloader,
    STFTProcessor,
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
from lingua.profiling import ProfilerArgs, maybe_run_profiler
# from lingua.tokenizer import build_tokenizer
from apps.AY2latent.transformer import (
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

import functools
logger = logging.getLogger()


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

    data: AudioDatasetArgs = field(default_factory=AudioDatasetArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    model: DecoderTransformerArgs = field(default_factory=DecoderTransformerArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)

    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
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
    


def process_batch_data(batch, current_step, data_processor, loss_weights, distill_model, resampler, load_distillation_model, distill_into_encoder, args_channel_loss_weighting, repa_into_encoder, repa_into_decoder):
    with torch.no_grad():
        if load_distillation_model:
            # with torch.no_grad():
            # distillation_signal = resampler(.squeeze(1))
            distillation_target = distill_model(input_values=batch['distill_16khz_decoder_audio'].squeeze(1).half(), output_hidden_states=True,).hidden_states[9].float()
        batch = data_processor.process(**batch)
        # with torch.no_grad():
        if load_distillation_model:
            distillation_target = F.adaptive_avg_pool1d(distillation_target.mT, batch['encoder_input'].shape[1]).mT.float().contiguous()
        if distill_into_encoder:
            batch['distill_target'] = distillation_target.detach()
        if args_channel_loss_weighting:
            batch['channel_loss_weighting'] = (loss_weights['channel_loss_weighting'] * min((current_step / 20000), 3)) + 1
            # loss_weights['channel_loss_weighting_max'] = batch['channel_loss_weighting'].max().item()
        if repa_into_encoder:
            batch['encoder_repa_target'] = distillation_target.detach()
        if repa_into_decoder:
            batch['decoder_repa_target'] = distillation_target.detach()
            loss_weights['decoder_repa_loss'] = loss_weights['decoder_repa_loss_original'] * min((current_step / 100000), 1)
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

    for source in args.data.data_csv_paths:
        # data_path = os.path.join(args.data.root_dir, source)
        assert os.path.exists(source), f"{source} doesn't exist"

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

    args.model.max_seqlen = int(args.data.sample_duration_seconds * args.data.sample_rate)

    if args.distributed.tp_size == 1:
        logger.warning(
            "Tensor parallelism has not been tested for a while, use at your own risk"
        )

    assert (
        args.probe_freq != args.profiling.mem_steps
    ), "Don't profile during probe step"
    assert (
        args.probe_freq != args.profiling.profile_steps
    ), "Don't profile during probe step"

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
        with torch.device("meta"):
            model = EncoderDecoder(args.model)
        logger.info("Model is built !")

        model_param_count = get_num_params(model)

        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=[],#build_fsdp_grouping_plan(args.model),
            # tp_parallelize=tp_parallelize,
            no_recompute_ops=get_no_recompute_ops(),
        )

        # Once we shard the model on different gpus we can actually initialize the model
        # First we create empty tensors of the correct shapes
        model = model.to_empty(device="cuda")
        # Then we init the model. Please make sure this function initializes *ALL* parameters
        # and buffers, otherwise you will have random values in the unitialized tensors
        # which will silently fail (give nan gradients for example)

        if args.checkpoint.init_ckpt_path:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(args.model.seed)
                model.init_weights()
            check_model_value_range(model, range=10.0, std=1.0)
            logger.info(f"!!!! Loading initial model from {args.checkpoint.init_ckpt_path} !!!! \n\n")
            load_from_checkpoint(args.checkpoint.init_ckpt_path, model, model_key="model") # Put model_key="" if its directly the model checkpoint
            
            model.encoder.rope_embeddings.reset_parameters() # For RoPe initialization since it's a buffer it might not be loaded
            model.decoder.rope_embeddings.reset_parameters() # For RoPe initialization since it's a buffer it might not be loaded
            model.decoder.t_embedder.reset_parameters(args.model.init_base_std)
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
        torch.cuda.manual_seed(rank_seed)

        logger.info(f"Setting torch seed to {rank_seed} for rank {dp_rank}")
        
        # Also make numpy and random seeds unique per rank
        np.random.seed(rank_seed)
        random.seed(rank_seed)

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
        args.data.load_csv()
        print("Entering create dataloader on rank", dp_rank)
        data_loader = create_dataloader(args.data, args.seed, dp_rank)
        print("Finishing create dataloader on rank", dp_rank)

        epoch = 0
        def make_batch_iterator(dataloader):
            nonlocal epoch
            dataloader.sampler.set_epoch(epoch)
            print("Creating batch iterator of dataloader with length", len(dataloader), "and dataset of length", len(dataloader.dataset))
            while True:
                epoch += 1
                logger.info(f"Starting epoch: {epoch}")
                for batch in dataloader:
                    yield batch

                dataloader.sampler.set_epoch(epoch)
                print("Finished epoch", epoch)

        batch_iterator = make_batch_iterator(data_loader)
        print("Entering create batch iterator on rank", dp_rank)

        torch_profiler = context_stack.enter_context(
            maybe_run_profiler(args.dump_dir, model, args.profiling)
        )

        #make sure all model parameters require gradients
        for p in model.parameters():
            p.requires_grad = True

        # model.encoder.init_generator(torch.cuda.current_device())

        data_processor = STFTProcessor(args.data).to(torch.cuda.current_device())

        loss_weights = None

        loss_weights = {"mmd": args.encoder_mmd_weight, "encoder_repa_loss": args.encoder_repa_weight,  "encoder_distill": args.encoder_distill_weight,
                       "decoder_repa_loss_original": args.decoder_repa_weight, "decoder_rf_loss": args.decoder_loss_weight}
        
        if args.channel_loss_weighting and type(data_processor.global_sigma) == torch.Tensor:
            with torch.inference_mode():
                eps = 1e-6
                # loss_weights = 1.0 / (data_processor.global_sigma**2 + eps)
                # loss_weights = loss_weights.unsqueeze(0).unsqueeze(0)
                channel_loss_weighting = 1.0 / (data_processor.global_sigma + eps)
                #put it to 0-1 range
                channel_loss_weighting = (((channel_loss_weighting - channel_loss_weighting.min()) / (channel_loss_weighting.max() - channel_loss_weighting.min())))


                channel_loss_weighting = channel_loss_weighting.unsqueeze(0).unsqueeze(0)
                #set require_grad to none
                channel_loss_weighting.requires_grad = False

                loss_weights['channel_loss_weighting'] = channel_loss_weighting

        if args.load_distillation_model:
            import torchaudio
            from transformers import HubertModel

            distill_model = HubertModel.from_pretrained("utter-project/mHuBERT-147", attn_implementation="flash_attention_2", torch_dtype = torch.float16).eval()
            distill_model = distill_model.to(torch.cuda.current_device())
            distill_model = torch.compile(distill_model)
            for p in distill_model.parameters():
                p.requires_grad = False
            resampler = torchaudio.transforms.Resample(orig_freq=args.data.sample_rate, new_freq=16000,).to(torch.cuda.current_device())

            logger.info("Created Hubert distillation model. Resampler initialized with original frequency: {} and new frequency: 16000".format(args.data.sample_rate))

        #partial process_batch_data with the stuff we have
        process_batch_data_compiled = functools.partial(
            process_batch_data,
            data_processor=data_processor,
            loss_weights=loss_weights,
            distill_model=distill_model if args.load_distillation_model else None,
            resampler=resampler if args.load_distillation_model else None,
            load_distillation_model=args.load_distillation_model,
            distill_into_encoder=args.distill_into_encoder,
            args_channel_loss_weighting=args.channel_loss_weighting,
            repa_into_encoder=args.repa_into_encoder,
            repa_into_decoder=args.repa_into_decoder,
        )
        process_batch_data_compiled = torch.compile(process_batch_data_compiled,)# mode="reduce-overhead", fullgraph=True)

        # Compile optimizer step
        @torch.compile()
        def optimizer_do_step():
            optimizer.step()
            optimizer.zero_grad()

        def fwd_step(batch):
            batch, loss_weights_batch = process_batch_data_compiled(batch, train_state.step)
            _, enc_loss, dec_loss = model(**batch)
            loss = sum(v * loss_weights_batch[k] for k, v in enc_loss.items()) + sum(v * loss_weights_batch[k] for k, v in dec_loss.items())
            if args.grad_acc_steps > 1:
                model.set_requires_gradient_sync(train_state.acc_step == 0)
            loss = loss / args.grad_acc_steps
            loss.backward()
            grad_norm = -1.0
            if train_state.acc_step == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.optim.clip, foreach=True
                )

                grad_norm = (
                    grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm
                ).item()

                optimizer_do_step()
                # optimizer.step()
                scheduler.step()
                # optimizer.zero_grad()
                train_state.step += 1

            return loss.item(), grad_norm

        nwords_since_last_log = 0
        time_last_log = timer()
        gc.collect()
        #clear cuda cache
        torch.cuda.empty_cache()
        if hasattr(optimizer, "train"):
            optimizer.train()
        while train_state.step < args.steps:
            # We constrain train_state.acc_step to be in range 0 to args.grad_acc_steps - 1
            train_state.acc_step += 1
            train_state.acc_step = train_state.acc_step % args.grad_acc_steps


            # get batch
            curr_lr = float(optimizer.param_groups[0]["lr"])
            data_load_start = timer()
            # batch = None
            # while batch is None:
            #     try:
            #         batch = next(batch_iterator)
            #     except Exception as e:
            #         print("The dataloader timed out lmfao", e)
            #         continue
            try:
                batch = next(batch_iterator)
            except Exception as e:
                print("The dataloader fucking died", e)
                data_loader = create_dataloader(args.data, args.seed+epoch+1, dp_rank)
                batch_iterator = make_batch_iterator(data_loader)
                batch = next(batch_iterator)

            #batch is a dictionary with keys: encoder_stft, decoder_input_stft, target, t
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}

            # batch = process_batch_data(args, train_state, data_processor, loss_weights, channel_loss_weighting, distill_model, resampler, batch)
            batch, loss_weights_batch = process_batch_data_compiled(batch, train_state.step)

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

            # #check for NaNs in the input
            # if torch.isnan(encoder_stft).any():
            #     logger.error("NaNs in encoder_stft")
            # if torch.isnan(decoder_input_stft).any():
            #     logger.error("NaNs in decoder_input_stft")
            # if torch.isnan(target).any():
            #     logger.error("NaNs in target")
            # if torch.isnan(t).any():
            #     logger.error("NaNs in t")


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

            _, enc_loss, dec_loss = model(**batch)
            # print("Encoder shit", enc_loss)
            # print("Decoder shit", dec_loss)
            # enc_loss and dec_loss are both dicts then do weighted sum by matching keys against loss_weights
            loss = sum(v * loss_weights_batch[k] for k, v in enc_loss.items()) + sum(v * loss_weights_batch[k] for k, v in dec_loss.items())

            if args.grad_acc_steps > 1:
                model.set_requires_gradient_sync(train_state.acc_step == 0)

            # We scale loss with grad_acc_steps so the gradient is the same
            # regardless of grad_acc_steps
            loss = loss / args.grad_acc_steps

            # backward on scaled loss to create scaled gradients
            loss.backward()
            # For logging we undo that scaling
            loss = loss.detach() * args.grad_acc_steps

            # optimizer step
            grad_norm = -1.0
            if train_state.acc_step == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.optim.clip, foreach=True
                )

                grad_norm = (
                    grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm
                ).item()

                optimizer_do_step()
                # optimizer.step()
                scheduler.step()
                # optimizer.zero_grad()
                train_state.step += 1

            # updates the scale for next iteration
            # training iteration complete
            end_timer.record()

            torch.cuda.synchronize()

            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

            # # if profiler is active
            # if torch_profiler:
            #     xformers.profiler.step()

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
                tokens_per_gpu = (
                    total_acc_steps * args.data.batch_size * batch['time_masks'].sum().item() #args.data.seq_len
                )
                total_tokens = dp_degree * tokens_per_gpu
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
                metrics = flatten_dict(
                    {
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
                            "total_tokens": total_tokens,
                        },
                        "memory": gpu_mem_stats._asdict(),
                    },
                    sep="/",
                )

                to_sync = {}
                to_sync["loss/out"] = loss.item()
                # to_sync["dec_loss/out"] = dec_loss.item()
                for k, v in dec_loss.items():
                    to_sync[f"dec_loss/{k}"] = v.mean().item()
                for k, v in enc_loss.items():
                    to_sync[f"enc_loss/{k}"] = v.mean().item()
                metrics.update(dist_mean_dict(to_sync))

                #add loss weights to metrics
                # metrics.update({'loss_weights/'+k: v for k, v in loss_weights.items()})

                if get_is_master():
                    metric_logger.log(metrics)

                gpu_memory_monitor.reset_peak_stats()
                nwords_since_last_log = 0
                time_last_log = timer()
                logger.info(
                    f"step: {train_state.step}"
                    # f"  acc: {train_state.acc_step}"
                    f"  loss: {round(loss.item(),4):>7}"
                    f"  dec_loss: {' '.join([f'{k}: {round(v.mean().item(),4):>7}' for k, v in dec_loss.items()])}" # log dec losses separately
                    f"  enc_loss: {' '.join([f'{k}: {round(v.mean().item(),4):>7}' for k, v in enc_loss.items()])}"
                    f"  grad: {grad_norm:.2e}"
                    # f"  flops: {FLOPS:.2e}"
                    # f"  wps: {wps:.2e}"
                    f"  iter: {curr_iter_time:>7}"
                    f"  data: {data_load_time:>5}"
                    f"  lr: {curr_lr:.2e}"
                    f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                    f"  pow: {gpu_mem_stats.power_draw/1000} W"
                )

            saved = False
            if every_n_steps(
                train_state, args.checkpoint.dump.every, acc_step=0
            ) or every_n_steps(train_state, args.checkpoint.eval.every, acc_step=0):
                #check if optimizer has .eval() method
                if hasattr(optimizer, "eval"):
                    optimizer.eval()

                # if loss_weights['decoder_repa_loss'] != args.decoder_repa_weight:
                #     loss_weights['decoder_repa_loss'] = args.decoder_repa_weight
                #     logger.info("Setting decoder repa loss to {}".format(args.decoder_repa_weight))



                original_csv_contents = args.data.csv_contents
                args.data.csv_contents = []

                saved = checkpoint.save(
                    model,
                    optimizer,
                    train_state,
                    args,
                    device_mesh=world_mesh,
                )
                args.data.csv_contents = original_csv_contents

                if hasattr(optimizer, "eval"):
                    optimizer.train()

            if args.eval is not None and every_n_steps(
                train_state, args.checkpoint.eval.every, acc_step=0
            ):
                if hasattr(optimizer, "eval"):
                    optimizer.eval()

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

            if preemption_flag["flag"]:
                if not saved:
                    if hasattr(optimizer, "eval"):
                        optimizer.eval()
                    original_csv_contents = args.data.csv_contents
                    args.data.csv_contents = []
                    checkpoint.save(
                        model,
                        optimizer,
                        train_state,
                        args,
                        device_mesh=world_mesh,
                    )
                    args.data.csv_contents = original_csv_contents
                    if hasattr(optimizer, "eval"):
                        optimizer.train()
                requeue_slurm_job()
                sys.exit(0)

    if not saved:
        if hasattr(optimizer, "eval"):
            optimizer.eval()
        original_csv_contents = args.data.csv_contents
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
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(TrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    # print(cfg)

    train(cfg)


if __name__ == "__main__":
    main()
