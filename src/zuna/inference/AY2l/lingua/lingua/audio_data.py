# audio_dataset.py
import os
import logging
import contextlib
import numpy as np

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Iterator, TypedDict

import torch
import torch.utils.data as td
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

# Import the relevant AudioTools classes/functions
# (Assuming your local structure or installed package)
from audiotools.data.datasets import AudioDataset, AudioLoader
from audiotools import transforms as atfm

logger = logging.getLogger(__name__)

###############################################################################
# ASYNC HELPER FUNCTIONS (same pattern as data.py)
###############################################################################

class StopFetch(Exception):
    """Custom exception to stop fetching data in the producer process."""
    pass

def feed_buffer(queue: mp.Queue, stop_event: mp.synchronize.Event, iterator_builder):
    """
    Producer function that pulls items from `iterator_builder()` and puts
    them into a multiprocessing queue, respecting a stop event.
    """
    with iterator_builder() as iterator:
        try:
            for item in iterator:
                # If the consumer or main process told us to stop, break
                if stop_event.is_set():
                    break
                # Keep trying to enqueue until success or forced stop
                while not stop_event.is_set():
                    try:
                        queue.put(item, timeout=0.1)
                        break
                    except mp.queues.Full:
                        pass
        except StopFetch:
            pass

def consume_buffer(producer: mp.Process, queue: mp.Queue):
    """
    Consumer generator that yields data from the queue. If the producer dies,
    we raise an error.
    """
    while producer.exitcode is None:
        try:
            yield queue.get(timeout=0.1)
        except mp.queues.Empty:
            pass
    raise RuntimeError(
        "Async dataloader: Producer process exited unexpectedly. "
        "Check logs for possible errors in the producer."
    )

@contextlib.contextmanager
def async_iterator(buffer_size: int, iterator_builder):
    """
    Context manager that launches a producer process that fetches items from
    `iterator_builder()` and pushes them into a queue. The consumer yields them.
    """
    queue = mp.Queue(maxsize=buffer_size)
    stop_event = mp.Event()
    producer = mp.Process(target=feed_buffer, args=(queue, stop_event, iterator_builder))
    logger.info("Starting async audio data loader process...")
    producer.start()

    consumer = consume_buffer(producer, queue)
    try:
        yield consumer
    finally:
        # Once we're done, signal the producer to stop
        stop_event.set()
        producer.join(timeout=0.2)
        if producer.exitcode is None:
            logger.info(f"Killing async data process (pid={producer.pid})...")
            producer.kill()
        else:
            logger.info(
                f"Async data process (pid={producer.pid}) exited with code {producer.exitcode}"
            )
        logger.info("Async audio data loader cleaned up.")


###############################################################################
# TYPED DICT(S) FOR STATE
###############################################################################

class AudioPrefetchState(TypedDict):
    """
    Minimal typed dict storing any info needed to reconstruct/resume the
    dataset/dataloader. In this example, we simply store:
      - dataset reference or configuration
      - rank / world_size (for distributed)
      - seed
      - iteration index, etc. (if we need explicit iteration states)
    Here we keep it extremely simple, but you can expand as needed.
    """
    dataset: Optional[AudioDataset]
    sampler: Optional[td.Sampler]
    dataloader: Optional[td.DataLoader]
    current_epoch: int
    # Potentially store random states, offsets, etc.


###############################################################################
# AUDIO ARGS (ANALOGOUS TO DataArgs IN data.py)
###############################################################################

@dataclass
class AudioDataArgs:
    """
    This mirrors the shape of DataArgs but for audio usage. You can
    rename or remove unneeded fields. We keep some the same for interface
    compatibility with train.py if needed.
    """
    # Original style fields:
    root_dir: Optional[str] = None
    sources: Dict[str, float] = field(default_factory=dict)

    batch_size: int = 2
    seed: int = 42

    # Unused in audio, but kept for interface parity
    seq_len: int = 16000       # If you want a "window" of audio frames
    n_views: int = 1           # Not typically used for audio, but left for parity
    load_async: bool = True
    prefetch_size: int = 64

    # Audio-specific fields:
    sample_rate: int = 16000
    duration: float = 1.0
    num_channels: int = 1
    shuffle: bool = True

    # Example transform toggles:
    augment_prob: float = 1.0
    # This is just to show how you'd pass transform specs, you can expand.
    # e.g. reverb_prob, eq_prob, etc.


###############################################################################
# HELPER: CREATE AUDIO DATASET
###############################################################################

def build_audiotools_dataset(
    args: AudioDataArgs,
    rank: int,
    world_size: int,
) -> AudioDataset:
    """
    Build an AudioDataset that merges all sources from `args.sources`.
    Each key in `args.sources` is some path or CSV folder, each value
    is a weight. In a typical usage, you'd combine them or do a simple
    random pick. For simplicity, let's combine them in a single dataset.
    
    Example usage of AudioLoader & AudioDataset:
      loader = AudioLoader(sources=["my/folder1", "my/folder2", ...])
      dataset = AudioDataset(loader, sample_rate=..., duration=..., ...)
    """
    # Flatten all source paths into a single list to pass to one AudioLoader
    # Weighted sampling inside AudioDataset is possible but not standard;
    # you could also create a ConcatDataset if you prefer. For brevity:
    all_paths = []
    for src_dir, weight in args.sources.items():
        # In a typical text pipeline, `weight` was used for multi-choice sampling,
        # but we'll just replicate each folder int(weight) times or similar:
        # For a better approach: implement your custom WeightedSampling inside AudioDataset.
        repeat_count = max(int(round(weight)), 1)
        all_paths.extend([src_dir] * repeat_count)

    if len(all_paths) == 0:
        logger.warning("No audio sources found in args.sources. Defaulting to empty dataset.")
        all_paths = [""]

    # Build an AudioLoader. Provide any transforms you want
    loader = AudioLoader(
        sources=all_paths,
        transform=None,     # We'll attach transformations in the dataset or later
        relative_path=args.root_dir or "",
        shuffle=args.shuffle,
        shuffle_state=args.seed + rank,  # keep it rank-dependent
    )

    # In AudioTools, AudioDataset can align or not. We'll keep it simple.
    dataset = AudioDataset(
        loaders=loader,
        sample_rate=args.sample_rate,
        duration=args.duration,
        num_channels=args.num_channels,
        # If you wanted to do multi-track alignment or re-loudness cutoff, pass them here:
        loudness_cutoff=-40,
        aligned=False,
        shuffle_loaders=args.shuffle,
        without_replacement=False,  # or True, depending on your preference
        n_examples=10_000_000,      # Large number => effectively "infinite"
    )
    return dataset


def infinite_dataloader(dataloader: td.DataLoader) -> Iterator[Any]:
    """
    Given a PyTorch DataLoader, yields batches in an infinite loop.
    This is akin to the text pipeline that never ends. If you want
    an epoch-based approach, remove or adapt.
    """
    while True:
        for batch in dataloader:
            yield batch


###############################################################################
# 1) init_dataloader_state_from_args
###############################################################################

def init_dataloader_state_from_args(
    args: AudioDataArgs,
    rank: int,
    world_size: int,
) -> AudioPrefetchState:
    """
    This is analogous to data.py's init_dataloader_state_from_args.
    We build the dataset, build a distributed sampler if needed, and store
    everything in a typed dict that the training loop can pass around or
    resume from.
    """
    # Build the dataset
    dataset = build_audiotools_dataset(args, rank, world_size)

    # Build sampler
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=args.shuffle,
            drop_last=False,
        )

    # We won't create the actual DataLoader yet. Or we can do so. Let's do it
    # so that the user can pass the state to build_dataloader_from_args later.
    dataloader = td.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None and args.shuffle),
        collate_fn=AudioDataset.collate,
        num_workers=4,     # Adjust as needed
        pin_memory=True,
        drop_last=False,
    )

    # For advanced usage, we might store random generator states, epoch, etc.
    # Keep it minimal here:
    state: AudioPrefetchState = {
        "dataset": dataset,
        "sampler": sampler,
        "dataloader": dataloader,
        "current_epoch": 0,
    }
    return state


###############################################################################
# 2) build_dataloader_from_args
###############################################################################

@contextlib.contextmanager
def build_dataloader_from_args(
    args: AudioDataArgs,
    state: Optional[AudioPrefetchState] = None,
) -> Iterator[Any]:
    """
    Analogous to data.py's build_dataloader_from_args. 
    Returns a context manager that yields an *iterator* over audio batches.
    
    If load_async=True, we wrap in the async producer/consumer pipeline 
    from data.py, otherwise we just yield from an infinite generator.
    """
    if state is None:
        raise ValueError(
            "No state provided to build_dataloader_from_args. "
            "Call init_dataloader_state_from_args(...) first."
        )

    dataset = state["dataset"]
    sampler = state["sampler"]
    dataloader = state["dataloader"]

    # If distributed sampler and we're doing epoch-based, set epoch here
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(state["current_epoch"])

    # Create a "repeating" iterator over the DataLoader
    def iterator_builder():
        return contextlib.nullcontext(infinite_dataloader(dataloader))

    if args.load_async:
        # We'll use the same pattern from data.py
        # buffer_size can be e.g. args.prefetch_size
        with async_iterator(buffer_size=args.prefetch_size, iterator_builder=iterator_builder) as gen:
            yield gen
    else:
        # Normal blocking iteration
        with iterator_builder() as gen:
            yield gen

    # If you track epochs, you might increment it here
    if sampler is not None and hasattr(sampler, "set_epoch"):
        state["current_epoch"] += 1
