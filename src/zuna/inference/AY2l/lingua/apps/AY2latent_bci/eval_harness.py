"""Shared fixed-eval-set harness (CLODE).

Both the in-training eval (train_compile_fwd.py) and the standalone plotter
(eeg_eval.py) import from here so they can NEVER drift, and so the samples
eeg_eval plots are literally the samples behind the training curve.

Design (see eval_harness_clode.md):
  * ONE ordered global pool of `num_batches` held-out samples, built ONCE and
    persisted to disk. Every rank and the 1-GPU plotter load the identical file.
  * FIXED total (world-size-invariant). Sharded across ranks with a strided
    partition; metrics all-reduced as (sum, count) so uneven shards are fine.
  * Per-sample sampler seed keyed by the sample's GLOBAL pool index (not rank,
    not shard position) -> every reconstruction is byte-identical across any
    world size and in the standalone plotter.
"""

import os
import torch
import torch.distributed as dist


# ---- keys yielded by make_batch_iterator (train_compile_fwd.py:688) ----------
# Kept here so the frozen pool is renamed/normalized IDENTICALLY to the live
# training/eval stream, independent of which caller built it.
_YIELD_KEYS = (
    "eeg_signal", "chan_pos", "chan_pos_discrete", "chan_id", "t_coarse",
    "token_dropout", "seq_lens", "max_tc", "pad_mask", "dataset_id",
)


def _normalize_and_select(batch, data_args):
    """Apply the same eeg_signal norm/clip + key-renaming that make_batch_iterator
    does, then move everything to CPU so it can be pickled to disk."""
    eeg = batch["eeg_signal"] / data_args.data_norm
    if data_args.data_clip is not None:
        eeg = eeg.clamp(min=-data_args.data_clip, max=data_args.data_clip)
    out = {k: batch[k] for k in _YIELD_KEYS}
    out["eeg_signal"] = eeg
    out["idx"] = batch["ids"]  # make_batch_iterator renames ids -> idx
    return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in out.items()}


def fixed_eval_cache_path(data_args, seed):
    """Canonical, RUN-INDEPENDENT path keyed by (data_dir, seed, N).

    Because it is derived only from the data config, the training run and the
    standalone plotter compute the SAME path for the SAME eval config, so
    whichever runs first builds the file and the other just loads it. Override
    the directory with `data_args.fixed_eval_cache_dir` if set."""

    data_dir = data_args.data_dir.rstrip("/")
    tag = os.path.basename(data_dir)
    cache_dir = getattr(data_args, "fixed_eval_cache_dir", None) \
        or os.path.join(os.path.dirname(data_dir), "eval_cache")
    return os.path.join(cache_dir, f"frozen_eval_{tag}_seed{seed}_N{data_args.num_batches}_TPS{data_args.target_packed_seqlen}_DO{data_args.dropout_scheme}.pt")


def build_or_load_fixed_eval_set(data_args, seed, is_master, rank_for_build=0):
    """Return a list of `num_batches` CPU batch dicts, identical across ranks/runs.

    Master builds+saves iff the cache file is missing; all ranks then load it.
    A dist barrier makes non-master ranks wait for the write. Dropout masks are
    baked into each batch by the dataset worker, so replaying these frozen
    batches gives identical samples+masks every eval."""
    from apps.AY2latent_bci.eeg_data import create_dataloader_v2

    path = fixed_eval_cache_path(data_args, seed)

    if not os.path.exists(path) and is_master:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dl = create_dataloader_v2(data_args, seed, rank_for_build)
        pool = []
        while len(pool) < data_args.num_batches:            # loop the loader if a single
            for batch in dl:                      # pass is shorter than N
                pool.append(_normalize_and_select(batch, data_args))
                if len(pool) >= data_args.num_batches:
                    break
        tmp = path + ".tmp"
        torch.save(pool, tmp)
        os.replace(tmp, path)                     # atomic: readers never see a partial file
        del dl

    if dist.is_available() and dist.is_initialized():
        dist.barrier()                            # everyone waits for master's write

    return torch.load(path, map_location="cpu")


def shard_indices(num_batches, rank, world_size):
    """Strided partition of [0, num_batches): rank r handles r, r+W, r+2W, ...
    Uneven shards are fine because metrics are reduced as (sum, count)."""
    if world_size <= 1:
        return list(range(num_batches))
    return list(range(rank, num_batches, world_size))


def sample_noise_seed(base_seed, global_idx):
    """Deterministic sampler seed for pool position `global_idx`. Keyed by the
    GLOBAL index, so a given sample reconstructs identically regardless of which
    rank runs it or how many ranks there are."""
    return int(base_seed) + int(global_idx)


def allreduce_metric_means(local_sums, local_count, device):
    """local_sums: {name: summed_value_over_this_rank}, local_count: #samples on
    this rank. Returns ({name: global_mean}, global_count) after summing across
    ranks. World-size-invariant: mean = (sum of per-sample values) / (total N)."""
    keys = sorted(local_sums.keys())
    vec = torch.tensor(
        [float(local_sums[k]) for k in keys] + [float(local_count)],
        device=device, dtype=torch.float64,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    count = vec[-1].item()
    means = {k: (vec[i].item() / count if count > 0 else float("nan"))
             for i, k in enumerate(keys)}
    return means, int(count)
