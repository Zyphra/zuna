import contextlib
import math
import os
from collections.abc import Generator, Iterable
from datetime import timedelta

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

import matplotlib.pyplot as plt
from datetime import datetime
import torch


@torch.no_grad()
def clip_grad_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        pp_mesh: pipeline parallel device mesh. If not None, will reduce gradient norm across PP stages.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(
        grads, norm_type, error_if_nonfinite, foreach
    )

    # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
    # We can simply reduce the DTensor to get the total norm in this tensor's process group
    # and then convert it to a local tensor.
    # NOTE: It has two purposes:
    #       1. to make sure the total norm is computed correctly when PP is used (see below)
    #       2. to return a reduced total_norm tensor whose .item() would return the correct value
    if isinstance(total_norm, DTensor):
        # Will reach here if any non-PP parallelism is used.
        # If only using PP, total_norm will be a local tensor.

        # Remove FT replicate dimension if it exists.
        total_norm = total_norm.full_tensor()

    total_norm **= norm_type
    dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
    total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


def reconstruct_full_mask(mask):
        """
        Utilities for testing and visualizing attention masks in EEG transformer models.

        Functions:
        - reconstruct_full_mask: Converts sparse block attention mask to full dense mask
        - visualize_attention_mask: Creates and saves attention pattern visualization
        """

        # Get the block structure (260x260 with 1s for active blocks)
        block_structure = mask.to_dense()[0, 0]  # 260x260 sparse block pattern
        full_shape = mask.shape[-1]  # Get the full sequence length (33180)
        full_mask = torch.zeros(full_shape, full_shape, device=block_structure.device, dtype=torch.bool)

        # Reconstruct full mask from active blocks only
        block_size = mask.BLOCK_SIZE[0]  # 128
        active_blocks = torch.where(block_structure == 1)  # Get coordinates of active blocks

        for q_block, kv_block in zip(active_blocks[0], active_blocks[1]):
            # Fill in the 128x128 regions for active blocks
            q_start, q_end = q_block * block_size, min((q_block + 1) * block_size, full_shape)
            kv_start, kv_end = kv_block * block_size, min((kv_block + 1) * block_size, full_shape)

            # Use mask_mod to get actual within-block attention pattern (vectorized)
            q_indices = torch.arange(q_start, q_end, device=block_structure.device)
            kv_indices = torch.arange(kv_start, kv_end, device=block_structure.device)
            q_grid, kv_grid = torch.meshgrid(q_indices, kv_indices, indexing='ij')

            # Call mask_mod with flattened arrays for efficiency
            block_mask = mask.mask_mod(0, 0, q_grid.flatten(), kv_grid.flatten())
            block_mask = block_mask.reshape(q_end - q_start, kv_end - kv_start)

            full_mask[q_start:q_end, kv_start:kv_end] = block_mask

        return full_mask

def visualize_attention_mask(mask, sample_size=5000, title_suffix=""):
    """
    Plot the attention mask. 
    Attentino mask needs to be constructed using reconstruct_full_mask()
    """
    if mask is not None:
        # Reconstruct full mask from block structure
        full_mask = reconstruct_full_mask(mask)
        mask_2d = full_mask.cpu().numpy()

        # Create binary attention pattern plot
        plt.figure(figsize=(10, 10))
        # Show sample or full mask depending on size
        display_mask = mask_2d[:sample_size, :sample_size] if mask_2d.shape[0] > sample_size else mask_2d
        plt.imshow(display_mask, cmap='Blues', aspect='equal')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.title(f'Attention Mask')
        plt.colorbar(label='Attention Allowed')

        # Generate filename with timestamp and config values
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"figures/attention_mask/mask_{title_suffix}.png"

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Attention mask saved to {save_path}")

def plot_random_samples_in_grid(data, 
                                num_samples=100, 
                                grid_rows=10, 
                                grid_cols=10, 
                                save_path='figures/enc_out_samples_grid.png', 
                                title='100 Random Samples from encoder output'):
    """
    Plot 100 random samples from xxx in a 10x10 grid and save as PNG.
    """
    random_indices = torch.randperm(data.shape[0])[:num_samples].cpu().numpy()

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(20, 20))
    fig.suptitle(title, fontsize=16)

    for idx, ax in enumerate(axes.flat):
        sample_idx = random_indices[idx]
        sample = data[sample_idx, :].float().detach().cpu().numpy()
        ax.plot(sample)
        ax.set_title(f'S{sample_idx}', fontsize=6)
        ax.tick_params(labelsize=4)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()