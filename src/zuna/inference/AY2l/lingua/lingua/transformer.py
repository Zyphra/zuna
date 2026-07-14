# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from torch.nn import RMSNorm as TorchRMSNorm

# from xformers.ops import fmha, AttentionBias (CW)
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
    _mask_mod_signature,
)

import math

from lingua import probe
# flex_attention_comp = torch.compile(flex_attention, dynamic=True, mode='max-autotune') # (CW) - This is the default for training. For eval, we can drop the mode='max-autotune' to save time.
flex_attention_comp = torch.compile(flex_attention, dynamic=True)                        # (CW) - ??? For eval, we drop the mode='max-autotune' to save time. Will slow down training time by ~10-40% on attention.
# flex_attention_comp = torch.compile(flex_attention)
# flex_attention_comp = flex_attention


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#
# (CW) - moved all this generate_doc_mask stuff into AY2latent_bci/transformer.py 

# def lengths_to_start_ids(lengths):
#     doc_start = lengths.cumsum(0)
#     doc_start = doc_start.roll(1)
#     doc_start[0] = 0
#     return doc_start


# def lengths_to_local_ids(lengths):
#     assert lengths.ndim == 1
#     nb_seqs = lengths.size(0)
#     total_seqlen = lengths.sum()
#     # This gives the document id of each token
#     doc_id = torch.repeat_interleave(lengths)
#     # Compute document start for each document
#     doc_start = lengths_to_start_ids(lengths)
#     # Compute document start for each token
#     doc_start = doc_start[doc_id]
#     # Compute the position of each token within each document
#     tok_id = torch.arange(total_seqlen, device=lengths.device) - doc_start

#     return doc_id, tok_id


# def generate_doc_mask_mod(
#     mask_mod: _mask_mod_signature,
#     lengths: torch.Tensor,
#     kv_lengths: Optional[torch.Tensor] = None, # for cross-attn
# ) -> _mask_mod_signature:
#     """Generates mask mods that apply to inputs to flex attention in the sequence stacked
#     format.

#     Args:
#         mask_mod: The mask mod to apply to the documents
#         lengths: Lengths of each document

#     Note:
#         What is the sequence stacked format? When assembling batches of inputs, we
#         take multiple sequences and stack them together to form 1 large sequence. We then
#         use masking to ensure that the attention scores are only applied to tokens within
#         the same document.

#     Example:

#     - Square mask
#       doc_mask         lengths
#       a a b b b c c    2 3 2
#     a 1 0 0 0 0 0 0
#     a 1 1 0 0 0 0 0
#     b 0 0 1 0 0 0 0
#     b 0 0 1 1 0 0 0
#     b 0 0 1 1 1 0 0
#     c 0 0 0 0 0 1 0
#     c 0 0 0 0 0 1 1

#     """

#     kv_lengths = kv_lengths if kv_lengths is not None else lengths
#     q_document_id, q_token_id = lengths_to_local_ids(lengths)
#     kv_document_id, kv_token_id = lengths_to_local_ids(kv_lengths)
#     q_max_idx = lengths.sum() - 1
#     kv_max_idx = kv_lengths.sum() - 1

#     def doc_mask_mod(b, h, q_idx, kv_idx):        
#         q_idx_cap = torch.minimum(q_max_idx, q_idx)
#         kv_idx_cap = torch.minimum(kv_max_idx, kv_idx)
#         valid_idx = (q_idx <= q_max_idx) & (kv_idx <= kv_max_idx)
#         same_doc = q_document_id[q_idx_cap] == kv_document_id[kv_idx_cap]
#         q_logical = q_token_id[q_idx_cap]
#         kv_logical = kv_token_id[kv_idx_cap]
#         inner_mask = mask_mod(b, h, q_logical, kv_logical)

#         return same_doc & inner_mask & valid_idx

#     return doc_mask_mod

# (CW) - moved all this generate_doc_mask stuff into AY2latent_bci/transformer.py 
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



class InitStdFactor(Enum):
    DISABLED = "disabled"  # Init std is divided by 1.0
    GLOBAL_DEPTH = "global_depth"  # Init std is divided by sqrt(2*n_layers)
    CURRENT_DEPTH = "current_depth"  # Init std is divided by sqrt(2*depth)
    DIM_RATIO = "dim_ratio"  # Init std is divided by model_dim/4096


@dataclass
class BaseTransformerArgs:

    dim: int = 1024
    n_layers: int = 10
    head_dim: Optional[int] = None
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None

    ffn_dim_multiplier: Optional[float] = None

    multiple_of: int = 256

    norm_eps: float = 1e-5

    rope_theta: float = 10000.0
    rope_dim: int = 1 # 0 = NoPE, 1 = 1D-RoPE, 4 = 4D=RoPE.
    tok_idx_type: Optional[str] = "t_coarse"

    ape_dim: int = 0 # 0 = No-APE, 1 = 1D-APE on ch_id,
    ape_theta: float = 10000.0 

    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"

    max_seqlen: int = 1024


def cross_entropy(pred, target, **kwargs):
    return F.nll_loss(
        F.log_softmax(pred.flatten(end_dim=-2).float(), -1),
        target.flatten(end_dim=-1),
        **kwargs,
    )


def repeat_kv(x: torch.Tensor, n_rep: int, dim: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    assert dim == 2, "Only dim=2 is supported. Check the implementation for other dims."
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    cos, sin = freqs.cos(), freqs.sin()

    return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        seq_dim (int): Sequence dimension index.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """

    ndim = x.ndim
    assert 0 <= seq_dim < ndim
    assert freqs_cis.shape == (
        x.shape[seq_dim],
        x.shape[-3],
        2,
        2,
    ), f"freqs_cis vs x: {(freqs_cis.shape, x.shape)}. freqs_cis should be{(x.shape[seq_dim], x.shape[-3], 2, 2)}."
    shape = [
        d if i == seq_dim or i == ndim - 3 else 1 for i, d in enumerate(x.shape[:-2])
    ] + [2, 2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    seq_dim: int,
    freqs_cis: torch.Tensor,
    # rope_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:

    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)  # B S D -> B S D/2 1 2
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)  # B S D -> B S D/2 1 2

    freqs_cis = reshape_for_broadcast(
        freqs_cis, xq_, seq_dim
    ).float()  # S D/2 2 2 -> 1 S 1 D/2 2 2
    xq_out = (xq_ * freqs_cis).sum(5).flatten(3)
    xk_out = (xk_ * freqs_cis).sum(5).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)



# Rotary embedding as in xformer, see if torchtrain implementation is not better. Also might be useful to make it work with batch*seqlen collapsed.
class RotaryEmbedding(torch.nn.Module):
    """
    RotaryEmbedding Module
    """

    def __init__(self, theta: float, head_dim: int, max_seqlen: int = 1024, rope_dim: int = 1):
        super().__init__()

        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen
        self.rope_dim = rope_dim

        assert head_dim % rope_dim == 0, f"head_dim must be divisible by rope_dim, got {head_dim} and {rope_dim}"

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(dim=head_dim//rope_dim, end=max_seqlen, theta=theta),
            persistent=False,
        )

    def reset_parameters(self):
        self.freqs_cis[...] = precompute_freqs_cis(
            dim=self.head_dim//self.rope_dim, end=self.max_seqlen, theta=self.theta
        )

    def forward(
        self, seqlen: Optional[int] = None, tok_idx: Optional[torch.Tensor] = None
    ):
        """
        Return freqs_cis corresponding to consecutive seqlen positions or the corresponding tok_idx positions
        Args:
            seqlen (int): Contiguous sequence length
            tok_idx (torch.Tensor[int]): Position indices of each token. This overrides seqlen.

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Embedded input tensor and freqs_cis
        """

        tok_idx = None # HARDCODE (CW)! SEE NOTE BELOW. WILL USE SEQLEN PATH.   

        test = (seqlen is not None) or (tok_idx is not None)
        assert test, "Should provide atleast seqlen or tok_idx"
        if tok_idx is not None:
            return self.freqs_cis[tok_idx] # NOTE: THINK I DONT WANT TO INDEX WITH TOK_IDX HERE AND THEN AGAIN INSIDE ATTENTION.FORWARD - DOUBLE DOING
        elif seqlen is not None:
            return self.freqs_cis[0:seqlen]


def sinusoidal_pe(max_len: int, d_model: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Returns [max_len, d_model] with 
                        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
                        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = torch.zeros(max_len, d_model, device=device, dtype=dtype)
    position = torch.arange(max_len, device=device, dtype=dtype).unsqueeze(1)  # [max_len, 1]
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / d_model)
    )  # [d_model // 2]
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


class SinusoidalEmbedding(nn.Module):
    """
    This module implements a sinusoidal (absolute) position embedding -  a la Attention is all you need.

    Args:
        dim (int): The dimension of the input tensor.
        max_seqlen (int): The maximum sequence length.
        model_dtype (str): The data type of the model. Must match model_dtype in distributed.model_dtype. Defaults to "fp32".
    """

    def __init__(self, dim: int, max_seqlen: int, model_dtype: str = "fp32"):
        super().__init__()
        self.dim = dim
        self.max_seqlen = max_seqlen
        assert dim % 2 == 0, f"SinusoidalEmbedding requires even dimension: got {dim}"
        param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[model_dtype]
        self.register_buffer(
            "pe",
            sinusoidal_pe(max_len=max_seqlen, d_model=dim, device="cpu", dtype=param_dtype),
            persistent=True,
        )
        self.scale = nn.Parameter(torch.tensor([0.1])) #, dtype=param_dtype))

    def reset_parameters(self):
        self.pe[...] = sinusoidal_pe(
            max_len=self.max_seqlen, d_model=self.dim, device=self.pe.device, dtype=self.pe.dtype
        )
        self.scale.data.fill_(0.1)

    def forward(self, ch_idx: torch.Tensor):
        return self.pe[ch_idx] * self.scale


class ScaledSinusoidalEmbedding(nn.Module):
    """Fixed sinusoidal buffer; learnable per-dimension scaling (and optional global gate)."""
    def __init__(self, dim: int, max_seqlen: int, model_dtype: str = "fp32"):
        super().__init__()
        self.dim = dim
        self.max_seqlen = max_seqlen
        assert dim % 2 == 0, f"requires even dim, got {dim}"
        param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[model_dtype]
        self.register_buffer(
            "pe",
            sinusoidal_pe(max_len=max_seqlen, d_model=dim, device="cpu", dtype=param_dtype),
            persistent=True,
        )
        # Per-dimension scale; starts at 1 so behavior matches unscaled sinusoidal PE
        self.scale = nn.Parameter(0.1*torch.ones(dim)) #, dtype=param_dtype))
        # Optional extra global factor 
        self.gate = nn.Parameter(torch.tensor([0.1])) #, dtype=param_dtype))

    def reset_parameters(self):
        with torch.no_grad():
            self.pe[...] = sinusoidal_pe(
                max_len=self.max_seqlen,
                d_model=self.dim,
                device=self.pe.device,
                dtype=self.pe.dtype,
            )
        self.scale.data.fill_(0.1)
        self.gate.data.fill_(0.1)

    def forward(self, ch_idx: torch.Tensor) -> torch.Tensor:
        # pe[ch_idx]: [..., dim]; scale: [dim] broadcasts
        return self.pe[ch_idx] * self.scale * self.gate



class LearnedSinusoidalInitEmbedding(nn.Module):
    """Absolute PE: learnable [max_seqlen, dim], initialized from analytic sinusoidal_pe."""
    def __init__(self, dim: int, max_seqlen: int, model_dtype: str = "fp32"):
        super().__init__()
        self.dim = dim
        self.max_seqlen = max_seqlen
        assert dim % 2 == 0, f"requires even dim, got {dim}"
        param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[model_dtype]
        self.pe = nn.Parameter(torch.empty(max_seqlen, dim, dtype=param_dtype))
        # Optional: global scalar
        self.gate = nn.Parameter(torch.tensor([1.0])) #, dtype=param_dtype))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            init = sinusoidal_pe(
                max_len=self.max_seqlen,
                d_model=self.dim,
                device=self.pe.device,
                dtype=self.pe.dtype,
            )
            #Line Below Causes RuntimeError: aten.copy_.default got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!
            self.pe.copy_(init) # COME BACK TO THIS.
        self.gate.data.fill_(1.0)

    def forward(self, ch_idx: torch.Tensor) -> torch.Tensor:
        return self.pe[ch_idx] * self.gate


class RMSNorm_MohVersion(nn.Module):
    """
    NOT USING ANYMORE. REPLACING WITH TORCH.NN.RMSNORM
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6, channel_dim=-1):
        super().__init__()
        self.eps = eps
        self.channel_dim = channel_dim


        # print(f"Inside RMSNorm.__init__, {channel_dim=}, {dim=}, {eps=}")
        # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

        if channel_dim != -1: #channel_dim is the index of the channel dimension, dim is the number of channels. assume 4 dimensions.
            self.weight = nn.Parameter(torch.ones([1]*channel_dim + [dim] + [1]*(4-channel_dim-1))).float()
        else:
            self.weight = nn.Parameter(torch.ones(dim)).float()


    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt((x * x).mean(self.channel_dim, keepdim=True) + self.eps) # "rsqrt" is reciprocal of sqrt.

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = probe.log_stats(x, "resid")
        output = self._norm(x.float())
        print(f"Inside RMSNorm.forward, {self.channel_dim=}, {self.weight.abs().max()=}")
        return (output * self.weight.float()).type_as(x) 

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore





class RMSNorm(nn.Module):
    """Drop-in replacement: torch.nn.RMSNorm + fp32 norm + probe."""

    def __init__(self, dim: int, eps: float = 1e-6, channel_dim: int = -1):
        super().__init__()
        if channel_dim != -1:
            raise NotImplementedError(
                "channel_dim != -1 not supported with torch.nn.RMSNorm; "
                "use normalized_shape=(..., dim) instead."
            )
        self.eps = eps
        self.norm = TorchRMSNorm(dim, eps=eps, elementwise_affine=True, dtype=torch.float32)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = probe.log_stats(x, "resid")

        x_fp32 = x.float()
        out = F.rms_norm(
            x_fp32,
            self.norm.normalized_shape,
            self.norm.weight.float(),  # explicit fp32 for the kernel
            self.norm.eps,
        )
        return out.type_as(x)

        ## HERE WEIGHTS WERE STILL IN BF16.
        # # Policy: normalize in fp32, cast back
        # out = self.norm(x.float()).type_as(x)
        # # CW - temp debug.
        # print(f"Inside RMSNorm.forward, {self.norm.weight.dtype=}, {x.float().dtype=}, {x.dtype=}")
        # return out

    def reset_parameters(self):
        self.norm.reset_parameters()

    @property
    def weight(self):
        return self.norm.weight  # optional: for code that reads .weight directly




class TiedLinear(nn.Module):
    def __init__(self, tied_module: nn.Module) -> None:
        super().__init__()
        self.tied_module = tied_module
        if not hasattr(tied_module, "weight"):
            raise AttributeError(
                "Provided module does not have attribute 'weight'. Please check your tied_module."
            )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.tied_module.weight)



class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        rope_dim: int,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.rope_dim = rope_dim

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads


        self.do_QK_norm = True
        if self.do_QK_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-5) #(CW) - 1e-5 is the default value in BaseTransformerArgs.norm_eps
            self.k_norm = RMSNorm(head_dim, eps=1e-5) 

        self.wq = nn.Linear(
            dim,
            n_heads * head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            n_heads * head_dim,
            dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask,  str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:

        # print("Inside lingua.transformer.Attention forward!!! ")
        # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)

        x = x.to(dtype=self.wq.weight.dtype) 

        # B S D
        bsz, seq_len, dim = x.shape
        xq = self.wq(x.view_as(x)) 
        xk = self.wk(x.view_as(x))
        xv = self.wv(x.view_as(x))

        output_shape = xq.shape
        # B S D -> B S H Dh         (where D = H*Dh)
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # DO QK_NORM HERE
        if self.do_QK_norm:
            # print(f"\n\nInside lingua.transformer.Attention forward, before qk_norm: {xq.shape=}, {xk.shape=}")
            # print(f"before qk_norm: {xq.abs().max()=}, {xk.abs().max()=}, {xq.norm(dim=-1).mean()=}, {xk.norm(dim=-1).mean()=}")
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
            # print(f"\n\nInside lingua.transformer.Attention forward, after qk_norm: {xq.shape=}, {xk.shape=}")
            # print(f"after qk_norm: {xq.abs().max()=}, {xk.abs().max()=},{xq.norm(dim=-1).mean()=}, {xk.norm(dim=-1).mean()=}")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)


        if self.rope_dim==0:
            # print("using NoPE in lingua.transformer.Attention.")
            pass
        elif self.rope_dim==1:

            # print(f"Inside attention block with 1d-RoPE: with \n \t{freq_cis.shape=}, \n \t{tok_idx.shape=}, \n \t{freq_cis[tok_idx].shape=}")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

            # Inside attention block with 1d-RoPE:
            #   freq_cis.shape            = [10, 32, 2, 2] = [max_seqlen, head_dim//2, 2, 2]
            #   tok_idx.shape             = [50400] = [seqlen]
            #   freq_cis[tok_idx].shape   = [50400, 32, 2, 2] = [seqlen, head_dim//2, 2, 2]

            if tok_idx is not None:
                xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[tok_idx])     # this edit mirrors what is inside RotaryEmbedding class. To use tok_idx
            else:
                xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])   # This is how it was. (SEEMS TO ASSUME WE ARE USING MAX_SEQLEN, NOT TOK_IDX)
        elif self.rope_dim==4:

            # print(f"Inside attention block with 4d-RoPE: \n \t{freq_cis.shape=}, \n \t{tok_idx.shape=}, \n \t{freq_cis[tok_idx].shape=}")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

            # Inside attention block with 4d-RoPE: 
            #     freq_cis.shape = [10, 8, 2, 2] = [max_seqlen, head_dim//(2*rope_dim), 2, 2] 
            #     tok_idx.shape = [50000, 4]) = [seqlen, rope_dim]
            #     freq_cis[tok_idx].shape = [50000, 4, 8, 2, 2] = [seqlen, rope_dim, head_dim//(2*rope_dim), 2, 2]

            # Build freqcis_4RoPE by indexing freq_cis with each dimension of tok_idx separately and concatenating
            # Cat along a new dimension to get [S, head_dim//2, 2, 2]
            freqcis_parts = []
            for i in range(self.rope_dim):
                freqcis_parts.append(freq_cis[tok_idx[:, i]])
            freqcis_4RoPE = torch.cat(freqcis_parts, dim=1)

            # Now apply 4D-axial-RoPE
            xq, xk = apply_rotary_emb(xq, xk, 1, freqcis_4RoPE) 


        else:
            print(f"I dont know how to handle {self.rope_dim=} inside lingua.transformer.Attention.forward")
            import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)


        # print(x, xq, xk, xv, freq_cis, tok_idx, seq_len)

        # This condition helps us be easily compatible with inference by adding a pluggable KVCache
        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        # print(f"Inside attention.forward, {mask=}") # (CW)

        if attn_impl == "flex_attention":
            assert mask is None or isinstance(mask, BlockMask)
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))


            output = flex_attention_comp(xq, xk, xv, block_mask=mask)
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

            # print(f"Inside lingua.transformer.Attention forward, after flex_attention_comp: {xq.shape=}, {xk.shape=}, {xv.shape=}, {output.shape=}")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)
            # print(f"Inside lingua.transformer.Attention forward, after flex_attention_comp: {xq.shape=}, {xk.shape=}, {xv.shape=}, {output.shape=}")

        elif attn_impl == "sdpa":
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            assert mask is None or isinstance(mask, (str, torch.Tensor))
            is_causal = (mask == "causal") if isinstance(mask, str) else False
            mask = mask if isinstance(mask, torch.Tensor) else None
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                is_causal=is_causal,
                attn_mask=mask,
            )
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D
        else:
            raise NotImplementedError(
                f"Attention implementation {attn_impl} not supported"
            )

        output = self.wo(output.reshape(output_shape))

        # print(f"INside Attention.forward, after attention, {output.shape=}, {output.dtype=}")
        # import IPython; print('\n\n Debug:'); IPython.embed(); import time;  time.sleep(0.3)

        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))

        for w in [self.wq, self.wk, self.wv]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )

        if self.do_QK_norm:
            self.q_norm.reset_parameters()
            self.k_norm.reset_parameters()


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        mp_size: int = 1,
    ):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % mp_size == 0

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # print("Inside lingua.transformer.FeedForward forward!!! ")
        # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)

        x = x.to(dtype=self.w1.weight.dtype)

        # B S D
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        output = self.w2(F.silu(x1) * x3)
        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        in_init_std = in_init_std
        out_init_std = out_init_std / factor
        for w in [self.w1, self.w3]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=in_init_std,
                a=-3 * in_init_std,
                b=3 * in_init_std,
            )
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )


class TransformerBlock(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()

        assert (args.head_dim is not None) or (
            args.n_heads is not None
        ), "Should specify at least head_dim or n_heads"
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // args.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads


        assert args.n_heads % self.n_kv_heads == 0
        assert args.dim % args.n_heads == 0

        self.attention = Attention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
            rope_dim=args.rope_dim,
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )

        self.do_sandwich_norm = True
        if self.do_sandwich_norm:
            self.attention_norm_post = RMSNorm(args.dim, eps=args.norm_eps)
            self.ffn_norm_post = RMSNorm(args.dim, eps=args.norm_eps)
        else:
            self.attention_norm_post = nn.Identity()
            self.ffn_norm_post = nn.Identity()
        
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        # print(f"Inside TransformerBlock __init__, ")
        # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

        # print(f"Insider TransformerBlock __init__, {args.dim=}, {args.norm_eps=}, ")

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask,  str]] = None,
        attn_impl: str = "sdpa",
        do_idx: Optional[torch.Tensor] = None,
        print_layerwise_activation_stats: bool = False,
    ) -> torch.Tensor:


        # print("\nInside TransformerBlock.forward:")

        if print_layerwise_activation_stats and do_idx is not None:

            print("DEPRECATED: FIX UP FLOATS AND INCLUDE SANDWICH NORM HERE IN TransformerBlock.forward")

            # # Print all the activation stats for the dropped and non-dropped tokens if do_idx is provided
            x_normed = self.attention_norm(x) # (CW)

            # print("\nInside TransformerBlock.forward with do_idx provided:")
            # import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

            print(f"\n\t Encoder attn_norm (drop-out): mean={x[:, do_idx, :].mean().item():.6f}, std={x[:, do_idx, :].std().item():.6f}", end=" --> ") # (CW)
            print(f"mean={x_normed[:, do_idx, :].mean().item():.6f}, std={x_normed[:, do_idx, :].std().item():.6f}") # (CW)
            print(f"\t Encoder attn_norm (non-drop): mean={x[:, ~do_idx, :].mean().item():.6f}, std={x[:, ~do_idx, :].std().item():.6f}", end=" --> ") # (CW)
            print(f"mean={x_normed[:, ~do_idx, :].mean().item():.6f}, std={x_normed[:, ~do_idx, :].std().item():.6f}") # (CW)
            h = + self.attention(                         # (CW) - lingua.transformer.Attention
                x_normed, # (CW)
                freq_cis,
                tok_idx=tok_idx,
                mask=mask, # (CW) - WORKS IF MASK=NONE.  FlexAttn BlockMask object does not get along with torch.compile()
                attn_impl=attn_impl,
            )
            h_normed = self.ffn_norm(h) # (CW)
            print(f"\n\t Encoder ffn_norm (drop-out): mean={h[:, do_idx, :].mean().item():.6f}, std={h[:, do_idx, :].std().item():.6f}", end=" --> ") # (CW)
            print(f"mean={h_normed[:, do_idx, :].mean().item():.6f}, std={h_normed[:, do_idx, :].std().item():.6f}") # (CW)
            print(f"\t Encoder ffn_norm (non-drop): mean={h[:, ~do_idx, :].mean().item():.6f}, std={h[:, ~do_idx, :].std().item():.6f}", end=" --> ") # (CW)
            print(f"mean={h_normed[:, ~do_idx, :].mean().item():.6f}, std={h_normed[:, ~do_idx, :].std().item():.6f}") # (CW)
            out = h + self.feed_forward(h_normed)  # (CW) - lingua.transformer.FeedForward

        else:

            # print(f"INside TransformerBlock.forward, before attention,  {x.dtype=}")
            # import IPython; print('\n\n Debug:'); IPython.embed(); import time;  time.sleep(0.3)

            # Attention Module:
            h = x.float() + self.attention_norm_post(
                self.attention(                                                                            # self-attention in BF16 / model_dtype
                    self.attention_norm(x.float()).to(dtype=self.attention.wq.weight.dtype),               # pre-norm in FP32
                    freq_cis,
                    tok_idx=tok_idx,
                    mask=mask, # (CW) - WORKS IF MASK=NONE.  FlexAttn BlockMask object does not get along with torch.compile()
                    attn_impl=attn_impl,
                ).float()                                                                                   # sandwich norm post-norm in FP32
            ).float()                                                                                       # residual in FP32

            # Feed Forward Module:
            out = h.float() + self.ffn_norm_post(
                self.feed_forward(                                                                          # FFN in BF16 / model_dtype
                    self.ffn_norm(h.float()).to(dtype=self.feed_forward.w1.weight.dtype)                    # do pre-norm in FP32
                ).float()                                                                                   # sandwich norm post-norm in FP32
            ).float()                                                                                       # residual in FP32     
        return out

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()

        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()


# class BaseTransformer(nn.Module):
#     def __init__(self, args: BaseTransformerArgs):
#         super().__init__()
#         self.dim = args.dim
#         self.init_base_std = args.init_base_std
#         self.init_std_factor = InitStdFactor(args.init_std_factor)
#         self.max_seqlen = args.max_seqlen
#         self.rope_embeddings = RotaryEmbedding(
#             theta=args.rope_theta,
#             head_dim=args.head_dim or args.dim // args.n_heads,
#             max_seqlen=args.max_seqlen,
#         )

#         self.layers = nn.ModuleList()
#         for _ in range(args.n_layers):
#             self.layers.append(TransformerBlock(args))

#         print("WRONG ONE BaseTransformer __init__")

#     def forward(
#         self,
#         h,
#         tok_idx: Optional[torch.Tensor] = None,
#         mask: Optional[Union[BlockMask,  str]] = None,
#         attn_impl: str = "sdpa",
#     ):
        
#         print("WRONG ONE BaseTransformer forward")

#         freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)

#         for i, layer in enumerate(self.layers):
#             h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
#         return h

#     def reset_parameters(self):
#         # Either use fixed base std or sqrt model dim
#         self.rope_embeddings.reset_parameters()

#     def init_weights(self):
#         self.reset_parameters()
#         for depth, layer in enumerate(self.layers):
#             factor = {
#                 InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
#                 InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
#                 InitStdFactor.DIM_RATIO: self.dim / 4096,
#                 InitStdFactor.DISABLED: 1.0,
#             }[self.init_std_factor]

#             layer.init_weights(self.init_base_std, factor)
