# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import contextlib
from dataclasses import dataclass
import logging
import os

import torch
import torch.distributed
import torch.profiler

from lingua.distributed import get_is_master

logger = logging.getLogger()


@dataclass
class ProfilerArgs:
    run: bool = False
    trace_folder: str = "profiling"
    profile_warmup: int = 100   # step at which recording starts
    profile_steps: int = 2      # how many steps to record


@contextlib.contextmanager
def maybe_run_profiler(dump_dir, module, config: ProfilerArgs):
    del module  # accepted for backward compat; unused
    if not config.run:
        yield None
        return

    trace_dir = os.path.join(dump_dir, config.trace_folder)
    if get_is_master() and not os.path.exists(trace_dir):
        os.makedirs(trace_dir)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    logger.info(f"Profiling active. Traces will be saved at {trace_dir}")

    cuda_available = torch.cuda.is_available()
    snapshot_path = os.path.join(trace_dir, "memory_snapshot.pickle")

    def on_trace_ready(prof):
        torch.profiler.tensorboard_trace_handler(trace_dir)(prof)
        if get_is_master():
            # high-signal operator summary sorted by total CUDA time; this is the
            # artifact to inspect/share first when hunting for efficiency wins
            table_path = os.path.join(trace_dir, "key_averages.txt")
            with open(table_path, "w") as f:
                f.write(prof.key_averages().table(
                    sort_by="cuda_time_total", row_limit=50
                ))
            logger.info(f"key_averages table saved to {table_path}")
        if cuda_available:
            torch.cuda.memory._dump_snapshot(snapshot_path)
            logger.info(f"CUDA memory snapshot saved to {snapshot_path}")

    if cuda_available:
        torch.cuda.memory._record_memory_history(max_entries=100_000)

    schedule = torch.profiler.schedule(
        wait=config.profile_warmup, warmup=0, active=config.profile_steps, repeat=1
    )
    try:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=schedule,
            on_trace_ready=on_trace_ready,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=True,
        ) as prof:
            yield prof
    finally:
        if cuda_available:
            torch.cuda.memory._record_memory_history(enabled=None)