# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from math import inf
from typing import Optional, Tuple, Union, List
from copy import deepcopy
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (
    create_block_mask, 
    BlockMask,
    _mask_mod_signature, 
    noop_mask,
)

torch._dynamo.config.capture_scalar_outputs = True 

from lingua.transformer import (
    RMSNorm,
    InitStdFactor,
    RotaryEmbedding,
    TransformerBlock,
)
from .xattn import DecoderBlock, FourierConditioner, DecoderArgs, AdaRMSNorm
from .conv_stem import CausalConv2DStem
from .bottlenecks import mmd_imq
from vector_quantize_pytorch import SimVQ, FSQ
import functools

def create_causal_mask(seqlen, attn_impl, sliding_window):
    if attn_impl == "sdpa":
        return "causal"
    elif attn_impl == "flex_attention":
        return create_block_mask(causal_mask, None, None, seqlen, seqlen)
    else:
        raise NotImplementedError(
            f"Attention {attn_impl} with {sliding_window} sliding window not implemented"
        )


def create_document_mask(lengths: torch.Tensor,
                         kv_lengths: Optional[torch.Tensor] = None, # for cross-attn
                         base_mask_mod: Optional[_mask_mod_signature] = None):
    """
    Create a document mask. Grabbing code from lingua.transformer
    """

    def generate_doc_mask_mod(
        mask_mod: _mask_mod_signature,
        lengths: torch.Tensor,
        kv_lengths: Optional[torch.Tensor] = None, # for cross-attn
    ) -> _mask_mod_signature:
        """Generates mask mods that apply to inputs to flex attention in the sequence stacked
        format.

        Args:
            mask_mod: The mask mod to apply to the documents
            lengths: Lengths of each document

        Note:
            What is the sequence stacked format? When assembling batches of inputs, we
            take multiple sequences and stack them together to form 1 large sequence. We then
            use masking to ensure that the attention scores are only applied to tokens within
            the same document.

        Example:

        - Square mask
        doc_mask         lengths
        a a b b b c c    2 3 2
        a 1 0 0 0 0 0 0
        a 1 1 0 0 0 0 0
        b 0 0 1 0 0 0 0
        b 0 0 1 1 0 0 0
        b 0 0 1 1 1 0 0
        c 0 0 0 0 0 1 0
        c 0 0 0 0 0 1 1

        """

        def lengths_to_start_ids(lengths):
            doc_start = lengths.cumsum(0)
            doc_start = doc_start.roll(1)
            doc_start[0] = 0
            return doc_start

        def lengths_to_local_ids(lengths):
            assert lengths.ndim == 1
            nb_seqs = lengths.size(0)
            total_seqlen = lengths.sum()
            # This gives the document id of each token
            doc_id = torch.repeat_interleave(lengths)
            # Compute document start for each document
            doc_start = lengths_to_start_ids(lengths)
            # Compute document start for each token
            doc_start = doc_start[doc_id]
            # Compute the position of each token within each document
            tok_id = torch.arange(total_seqlen, device=lengths.device) - doc_start
            return doc_id, tok_id

        kv_lengths = kv_lengths if kv_lengths is not None else lengths
        q_document_id, q_token_id = lengths_to_local_ids(lengths)
        kv_document_id, kv_token_id = lengths_to_local_ids(kv_lengths)
        q_max_idx = lengths.sum() - 1
        kv_max_idx = kv_lengths.sum() - 1

        def doc_mask_mod(b, h, q_idx, kv_idx):        
            q_idx_cap = torch.minimum(q_max_idx, q_idx)
            kv_idx_cap = torch.minimum(kv_max_idx, kv_idx)
            valid_idx = (q_idx <= q_max_idx) & (kv_idx <= kv_max_idx)
            same_doc = q_document_id[q_idx_cap] == kv_document_id[kv_idx_cap]
            q_logical = q_token_id[q_idx_cap]
            kv_logical = kv_token_id[kv_idx_cap]
            inner_mask = mask_mod(b, h, q_logical, kv_logical)
            return same_doc & inner_mask & valid_idx

        return doc_mask_mod

    if base_mask_mod is None:
        base_mask_mod = noop_mask

    doc_mask_mod = generate_doc_mask_mod(base_mask_mod, lengths)

    return create_block_mask(doc_mask_mod, None, None, lengths.sum().item(), lengths.sum().item())


def attention_flops_per_token(n_layers, seq_len, dim, causal):
    # Formula from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
    return 3.5 * (4 * n_layers * seq_len * dim // (2 if causal else 1))


def get_num_flop_per_token(
    num_non_embed_params: int, n_layers: int, dim: int, seq_len: int
) -> int:
    return 6 * num_non_embed_params + attention_flops_per_token(
        n_layers, seq_len, dim, True
    )


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def extract_non_registers(h: torch.Tensor, num_groups: int, original_seqlen: int = None, downsample_factor: int = None) -> torch.Tensor:
    """
    Extract non-register tokens from the output tensor.
    Args:
        h: Output tensor from transformer layers [B, interleaved_seqlen, D].
        num_groups: Number of groups used in interleaving.
        original_seqlen: The sequence length of the input *before* padding
                            and interleaving. Used to trim non-register tokens.

    Returns:
        non_registers: [B, original_seqlen, D]
    """
    bsz, seq_len, dim = h.shape
    # seq_len should be num_groups*(df+1)
    h = h.reshape(bsz, num_groups, downsample_factor + 1, dim)

    # Extract non-register tokens (indices 1 to df)
    non_registers = h[:, :, 1:, :]
    # Flatten back to sequence dimension
    padded_seqlen = num_groups * downsample_factor
    non_registers = non_registers.reshape(bsz, padded_seqlen, dim)
    # Trim back to the original sequence length, removing padding effects
    non_registers = non_registers[:, :original_seqlen, :]  # [B, original_seqlen, D]

    return non_registers.contiguous()

@torch.compile()
def huber_loss(target, logits, huber_c):
    return huber_c * (torch.sqrt((target - logits) ** 2 + huber_c**2) - huber_c)

@torch.compile()
def cosine_similarity_loss(input, target):
    return (1 - F.cosine_similarity(input, target, dim=-1).mean())

@torch.compile()
def huber_cosine_weighted(input, target, huber_c = 0.1):
    # Compute the Huber loss
    h_loss = huber_loss(input, target, huber_c).mean() * 0.5
    
    # Compute the cosine similarity
    cosine_sim = cosine_similarity_loss(input, target)
    
    # Combine the two losses
    combined_loss = h_loss + cosine_sim
    
    return combined_loss

@dataclass
class DecoderTransformerArgs(DecoderArgs):

    seed: int = 42

    weight_tying: bool = False
    sliding_window: int = 128
    xattn_sliding_window: int = 32
    input_dim: int = 64 

    decoder_encoder_dropout: float = 0.1
    decoder_timestep_dropout: float = 0.1

    encoder_sliding_window: int = 128
    encoder_input_dim: int = input_dim 
    encoder_output_dim: int = input_dim*2
    encoder_latent_downsample_factor: int = 2  
    encoder_hidden_dim: Optional[int] = None

    adaptive_loss_weighting: bool = False  
    num_fine_time_pts: int = 128
    dont_noise_chan_xyz: bool = False
    stft_global_sigma: Union[str, float] = 1.0 

    dropout_type: str =  "zero" # {"zero", "rand", "learnable"}

    bottleneck_type: str = "mmd"
    distill_output_dim: int = 0
    codebook_size: int = 1024
    levels: List[int] = field(default_factory=list)
    init_base_std: float = 0.02
    learnable_bias: bool = False

    huber_c: Optional[float] = None

    decoder_repa_index: float = float('inf')
    encoder_repa_index: float = float('inf')
    repa_dim: int = 1024
    repa_loss_fn: str = "cosine"

    compression_free_conv_stem: bool = False

    max_seqlen: int = 1024

class BaseTransformerDecoder(nn.Module):
    def __init__(self, args: DecoderTransformerArgs):
        super().__init__()

        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seqlen = args.max_seqlen
        self.rope_embeddings = RotaryEmbedding(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.max_seqlen,
            rope_dim=args.rope_dim,
        )

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(DecoderBlock(args))

        self.repa_index = args.decoder_repa_index
        if self.repa_index != inf:
            # self.repa_proj = nn.Linear(args.dim, args.repa_dim)
            self.repa_proj = nn.Sequential(
                nn.Linear(args.dim, args.repa_dim),
                nn.SiLU(),
                nn.Linear(args.repa_dim, args.repa_dim),
                nn.SiLU(),
                nn.Linear(args.repa_dim, args.repa_dim),
            )
            self.repa_norm = AdaRMSNorm(args.t_dim, args.dim, eps=args.norm_eps)
            self.repa_loss_fn = cosine_similarity_loss if args.repa_loss_fn == "cosine" else huber_cosine_weighted

    def forward(
        self,
        h,
        x_attended,
        t,
        tok_idx: Optional[torch.Tensor] = None,
        cross_tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask,  str]] = None,
        cross_attn_mask: Optional[Union[BlockMask,  str]] = None,
        attn_impl: str = "sdpa",
        repa_target: Optional[torch.Tensor] = None,
        do_idx: Optional[torch.Tensor] = None, # (CW)
    ):

        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)
        repa_loss = None

        for i, layer in enumerate(self.layers):     # all these layers are type 'xattn.DecoderBlock'
            h = layer(h,
                      x_attended,
                      t,
                      freq_cis,
                      tok_idx=tok_idx,
                      cross_tok_idx=cross_tok_idx,
                      self_attn_mask=mask,
                      cross_attn_mask=cross_attn_mask,
                      attn_impl=attn_impl,
                      do_idx=do_idx, # (CW)
            )

            if self.training and self.repa_index != inf and i == self.repa_index:
                repa_loss = self.repa_loss_fn(self.repa_proj(self.repa_norm(h, t)).float(), repa_target,)

        return h, repa_loss



    def reset_parameters(self):
        # Either use fixed base std or sqrt model dim
        self.rope_embeddings.reset_parameters()

    def init_weights(self):
        self.reset_parameters()
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(self.init_base_std, factor)

        # Add these lines for repa_proj initialization
        if self.repa_index != float('inf'):
            init_std = self.init_base_std or (self.dim ** (-0.5))
            self.repa_norm.reset_parameters() # Ensure repa_norm is also reset
            
            #now repa_proj is nn.Sequential, let's do it in a loop
            for i, layer in enumerate(self.repa_proj):
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(
                        layer.weight,
                        mean=0.0,
                        std=init_std,
                        a=-3 * init_std,
                        b=3 * init_std,
                    )
                    # nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            

class BaseTransformer(nn.Module):
    def __init__(self, args: DecoderTransformerArgs):
        super().__init__()

        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seqlen = args.max_seqlen
        self.rope_embeddings = RotaryEmbedding(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.max_seqlen,
            rope_dim=args.rope_dim,
        )

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))

        self.repa_index = args.encoder_repa_index
        if self.repa_index != inf:
            self.repa_proj = nn.Linear(args.dim, args.repa_dim)
            self.repa_norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.repa_loss_fn = cosine_similarity_loss if args.repa_loss_fn == "cosine" else huber_cosine_weighted


    def forward(
        self,
        h,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask,  str]] = None,
        attn_impl: str = "sdpa",
        repa_target: Optional[torch.Tensor] = None,
        do_idx: Optional[torch.Tensor] = None,
        **kwargs
    ):

        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen,
                                        tok_idx=tok_idx
        )
        repa_loss = None

        for i, layer in enumerate(self.layers):     # all these layers are type 'TransformerBlock'
            h = layer(h, 
                      freq_cis, 
                      tok_idx=tok_idx, 
                      mask=mask, 
                      attn_impl=attn_impl,
                      do_idx=do_idx,
            )

            if self.training and self.repa_index != inf and i == self.repa_index:
                repa_loss = cosine_similarity_loss(self.repa_proj(self.repa_norm(extract_non_registers(h, **kwargs))).float(), repa_target,)


        return h, repa_loss

    def reset_parameters(self):
        # Either use fixed base std or sqrt model dim
        self.rope_embeddings.reset_parameters()

    def init_weights(self):
        self.reset_parameters()
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(self.init_base_std, factor)

        # Add these lines for repa_proj initialization
        if self.repa_index != float('inf'):
             init_std = self.init_base_std or (self.dim ** (-0.5))

             nn.init.trunc_normal_(
                 self.repa_proj.weight,
                 mean=0.0,
                 std=init_std, # Use out_init_std based on repa_dim if desired, using self.dim for now
                 a=-3 * init_std,
                 b=3 * init_std,
             )
             if self.repa_proj.bias is not None:
                 nn.init.zeros_(self.repa_proj.bias)
             self.repa_norm.reset_parameters() # Ensure repa_norm is also reset

class DecoderTransformer(BaseTransformerDecoder):
    def __init__(self, args: DecoderTransformerArgs):
        super().__init__(args)
        self.weight_tying = False
        self.sliding_window = args.sliding_window
        self.xattn_sliding_window = args.xattn_sliding_window
        if args.huber_c is not None:
            self.huber_c = args.huber_c
        else:
            self.huber_c = None 

        self.tok_embeddings = nn.Linear(args.input_dim, args.dim,) 

        self.t_embedder = FourierConditioner(args.t_dim, std=args.init_base_std)

        self.encoder_proj = nn.Linear(args.encoder_output_dim, args.dim)

        self.norm = AdaRMSNorm(args.t_dim, args.dim, eps=args.norm_eps)

        self.output = nn.Linear(
            args.dim,
            args.input_dim, 
            bias=False,
        )
        self.init_base_std = args.init_base_std
        self.use_compression_free_conv_stem = False
        if args.compression_free_conv_stem:
            self.use_compression_free_conv_stem = True

            self.compression_free_conv_stem_input = CausalConv2DStem(
                input_features = args.input_dim, 
                hidden_channels = 32,
                activation = nn.SELU,
                compress_channels=False,
            )
            self.compression_free_conv_stem_output = CausalConv2DStem(
                input_features = args.input_dim,
                hidden_channels = 32,
                activation = nn.SELU,
                compress_channels=False,
            )

        self.adaptive_loss_weighting = args.adaptive_loss_weighting


    def forward(
        self,
        tokens: torch.Tensor,
        cross_attended: torch.Tensor,
        timeD: torch.Tensor,
        seq_lens: torch.Tensor,         # for document masking packed sequences in self-attention
        cross_seq_lens: torch.Tensor,   # for document masking packed sequences in cross-attention
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        cross_tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask,  torch.Tensor, str]] = None,
        cross_attn_mask: Optional[Union[BlockMask,  str]] = None,
        attn_impl: str = "flex_attention",
        time_masks: Optional[torch.Tensor] = None,
        channel_loss_weighting: Optional[torch.Tensor] = None, # [1, 1, input_dim*2]
        repa_target: Optional[torch.Tensor] = None,
        freq_masks: Optional[torch.Tensor] = None,
        do_idx: Optional[torch.Tensor] = None,
        print_layerwise_activation_stats: bool = False,
    ):

        tokens = tokens.squeeze(1)  

        bsz, seqlen, dim = tokens.shape
        _, cross_seqlen, _ = cross_attended.shape

        # Masking out channels that were set to all-zeros
        if self.training and freq_masks is not None:
            with torch.no_grad():
                tokens *= freq_masks
        
        if self.use_compression_free_conv_stem:
            tokens = self.compression_free_conv_stem_input(tokens)

        h = self.tok_embeddings(tokens)
        t = self.t_embedder(timeD)

        cross_attended = self.encoder_proj(cross_attended)


        # COMBINE SLIDING WINDOW MASK AND DOCUMENT MASK
        SLIDING_WINDOW = self.sliding_window
        def selfattn_sliding_window_func(b, h, q_idx, kv_idx):
            # Self-attention case
            return (q_idx - kv_idx).abs() <= SLIDING_WINDOW
        mask_mod_slide = selfattn_sliding_window_func
        mask = create_document_mask(lengths=seq_lens, base_mask_mod=mask_mod_slide)
        #
        SLIDING_WINDOW = self.xattn_sliding_window
        def crossattn_sliding_window_func(b, h, q_idx, kv_idx):
            # Cross-attention case
            center_k_idx = (q_idx * cross_seqlen) // seqlen
            return (kv_idx - center_k_idx).abs() <= SLIDING_WINDOW
        mask_mod_slide = crossattn_sliding_window_func
        cross_attn_mask = create_document_mask(lengths=seq_lens, kv_lengths=cross_seq_lens, base_mask_mod=mask_mod_slide)

        visualize_attention_masks = False
        if visualize_attention_masks:
            from .utils import visualize_attention_mask
            torch._dynamo.config.disable = True
            visualize_attention_mask(mask, title_suffix="decoder_self_attn")
            visualize_attention_mask(cross_attn_mask, title_suffix="decoder_cross_attn")
            torch._dynamo.config.disable = False


        if tok_idx is not None:
            if tok_idx.ndim==3 and tok_idx.shape[0]==1:
                tok_idx = tok_idx.squeeze().squeeze() # make it the right size for RoPE.

        if cross_tok_idx is not None:
            if cross_tok_idx.ndim==3 and cross_tok_idx.shape[0]==1:
                cross_tok_idx = cross_tok_idx.squeeze().squeeze() # make it the right size for RoPE.

        h, repa_loss = super().forward(h,
                                       cross_attended,
                                       t=t,
                                       tok_idx=tok_idx,
                                       cross_tok_idx=cross_tok_idx,
                                       mask=mask, 
                                       cross_attn_mask=cross_attn_mask, 
                                       attn_impl=attn_impl,
                                       repa_target=repa_target,
                                       do_idx=do_idx,
        )

        h_normed = self.norm(h, t) 

        if print_layerwise_activation_stats and do_idx is not None: 
            print(f"\nDecoder output norm: (drop-out) mean={h[:, do_idx, :].mean().item():.6f}, std={h[:, do_idx, :].std().item():.6f}", end=" --> ") # (CW)
            print(f"mean={h_normed[:, do_idx, :].mean().item():.6f}, std={h_normed[:, do_idx, :].std().item():.6f}") 
            print(f"Decoder output norm: (non-drop) mean={h[:, ~do_idx, :].mean().item():.6f}, std={h[:, ~do_idx, :].std().item():.6f}", end=" --> ") # (CW)
            print(f"mean={h_normed[:, ~do_idx, :].mean().item():.6f}, std={h_normed[:, ~do_idx, :].std().item():.6f}")

        logits = self.output(h_normed) 

        if self.use_compression_free_conv_stem:
            logits = self.compression_free_conv_stem_output(logits)

        losses = self.compute_losses(target, logits, time_masks, freq_masks, channel_loss_weighting)

        if repa_target is not None:
            losses["decoder_repa_loss"] = repa_loss#.mean()

        return logits, losses


    
    @torch.compile()
    def compute_losses(self, target, logits, time_masks, freq_masks, channel_loss_weighting):
        losses = {}

        if target is not None:
            if self.huber_c is None:
                batchwise_loss = F.mse_loss(target.float(), logits.float(), reduction="none") # shape = [B, T, C]
            else:
                batchwise_loss = huber_loss(target.float(), logits.float(), self.huber_c)

            #  Do Adaptive Loss Weighting - to boost loss from channels with small signals so we can better learn small signals.
            if self.adaptive_loss_weighting:
                ALW = batchwise_loss.detach().abs().mean(dim=2).unsqueeze(2) # shape = [B,C,1]
                batchwise_loss = batchwise_loss/(ALW + 1e-5)


            if channel_loss_weighting is not None:
                batchwise_loss = batchwise_loss * channel_loss_weighting

            if freq_masks is not None:
                batchwise_loss = (batchwise_loss * freq_masks).sum(dim=1) / (freq_masks.sum(dim=1) + 1e-6) # shape = [B,C,1]
            else:
                batchwise_loss = batchwise_loss.mean(dim=-1)

            if time_masks is not None:
                batchwise_loss = (batchwise_loss * time_masks).sum(dim=-1) / time_masks.sum(dim=-1)

            losses["decoder_rf_loss"] = batchwise_loss.mean()

        return losses


    def reset_parameters(self, init_std=None):
        # Either use fixed base std or sqrt model dim
        super().reset_parameters()
        if init_std is None:
            init_std = self.init_base_std or (self.dim ** (-0.5))

        self.norm.reset_parameters()
        self.t_embedder.reset_parameters(init_std)
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )

        nn.init.trunc_normal_(
            self.encoder_proj.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        if self.encoder_proj.bias is not None:
            nn.init.zeros_(self.encoder_proj.bias)

        nn.init.trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        if self.use_compression_free_conv_stem:
            self.compression_free_conv_stem_input.reset_parameters(init_std)
            self.compression_free_conv_stem_output.reset_parameters(init_std)
            
    def init_weights(self):
        super().init_weights()
        self.reset_parameters()

class EncoderTransformer(BaseTransformer):
    def __init__(self, args: DecoderTransformerArgs):
        args = deepcopy(args)
        args.dim = args.dim if args.encoder_hidden_dim is None else args.encoder_hidden_dim
        super().__init__(args)
        self.weight_tying = False
        self.sliding_window = args.encoder_sliding_window
        self.bottleneck_type = args.bottleneck_type
        self.downsample_factor = args.encoder_latent_downsample_factor
        self.distill = args.distill_output_dim != 0
        self.tok_embeddings = nn.Linear(args.encoder_input_dim, args.dim) 
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.registers = torch.nn.Parameter(torch.zeros(1, args.encoder_input_dim))
        self.dropout_type = args.dropout_type
        if self.dropout_type=="learnable":
            self.dropout_vec = torch.nn.Parameter(args.stft_global_sigma*torch.rand(1, args.encoder_input_dim, dtype=torch.float32)) #, device='cuda')) # rand init for learnable dropout vector # (CW)
        else:
            self.dropout_vec = None # If None, it will just use zeros for dropped out chans (rather than learnable vector).

        self.init_base_std = args.init_base_std
        self.output = nn.Linear(args.dim, args.encoder_output_dim, bias=False)

        if args.distill_output_dim != 0:
            self.distill_output = nn.Linear(
                args.dim,
                args.distill_output_dim,
                bias=True,
            )
            self.distill_norm = RMSNorm(args.dim, eps=args.norm_eps)

        if "sim" in args.bottleneck_type:
            self.quantizer = SimVQ(
                dim = args.encoder_output_dim,
                codebook_size = args.codebook_size,
                rotation_trick = True  # use rotation trick from Fifty et al.
            )
        elif "fsq" in args.bottleneck_type:
            self.quantizer = FSQ(
                levels = args.levels
            )

        self.use_compression_free_conv_stem = False
        if args.compression_free_conv_stem:
            self.use_compression_free_conv_stem = True

            self.compression_free_conv_stem_input = CausalConv2DStem(
                input_features = args.input_dim, # (CW) *2 was for STFT amplitude & phase
                hidden_channels = 32,
                activation = nn.SELU,
                compress_channels=False,
            )


    def _interleave_registers(self, x: torch.Tensor):
        """
        1) Pad `x` along the sequence dimension so it’s divisible by `self.downsample_factor`.
        2) Reshape into groups of length `df`.
        3) Insert a copy of `self.registers` in front of each group.
        4) Flatten back out.

        Returns:
            interleaved: [B, num_groups*(df+1), D]
            num_groups: int
        """

        bsz, seqlen, dim = x.shape
        df = self.downsample_factor

        # Number of groups
        num_groups = (seqlen + df - 1) // df
        new_seqlen = num_groups * df

        # Pad if needed
        if new_seqlen > seqlen:
            pad_len = new_seqlen - seqlen
            x = torch.cat([x, x.new_zeros(bsz, pad_len, dim)], dim=1)

        # Reshape to [B, num_groups, df, D]
        x = x.reshape(bsz, num_groups, df, dim)

        # Expand the single register => [B, num_groups, 1, D]
        regs = self.registers.expand(bsz, num_groups, -1).unsqueeze(2)

        # Cat the register in front of each group => [B, num_groups, df+1, D]
        x = torch.cat([regs, x], dim=2)

        # Flatten => [B, num_groups*(df+1), D]
        x = x.reshape(bsz, -1, dim).contiguous()
        return x, num_groups

    def _extract_registers_and_non_registers(
        self,
        h: torch.Tensor,
        num_groups: int,
        original_seqlen: int = None,                                # Added: original sequence length before padding
        return_non_registers: bool = False                          # Added: flag to return other tokens
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:    # Updated return type hint
        """
        Args:
            h: Output tensor from transformer layers [B, interleaved_seqlen, D].
            num_groups: Number of groups used in interleaving.
            original_seqlen: The sequence length of the input *before* padding
                             and interleaving. Used to trim non-register tokens.
            return_non_registers: If True, return both register and non-register
                                  tokens. Otherwise, return only register tokens.

        Returns:
            If return_non_registers is False:
                registers: [B, num_groups, D]
            If return_non_registers is True:
                registers: [B, num_groups, D]
                non_registers: [B, original_seqlen, D]
        """
        bsz, seq_len, dim = h.shape
        df = self.downsample_factor
        # seq_len should be num_groups*(df+1)
        h = h.reshape(bsz, num_groups, df + 1, dim)

        # The register is the first token in each group
        registers = h[:, :, 0, :]  # [B, num_groups, D]

        if not return_non_registers:
            return registers.contiguous(), None
        else:
            # Extract non-register tokens (indices 1 to df)
            non_registers = h[:, :, 1:, :] # [B, num_groups, df, D]
            # Flatten back to sequence dimension
            padded_seqlen = num_groups * df
            non_registers = non_registers.reshape(bsz, padded_seqlen, dim)
            # Trim back to the original sequence length, removing padding effects
            non_registers = non_registers[:, :original_seqlen, :] # [B, original_seqlen, D]

            return registers.contiguous(), non_registers.contiguous()
        


    def forward(
        self,
        token_values: torch.Tensor,
        seq_lens: torch.Tensor,
        distill_target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask,  torch.Tensor, str]] = None,
        attn_impl: str = "flex_attention",
        repa_target: Optional[torch.Tensor] = None,
        do_idx: Optional[torch.Tensor] = None,
        print_layerwise_activation_stats: bool = False,
    ): 
        
        _, orig_seqlen, _ = token_values.shape

        if self.use_compression_free_conv_stem:
            token_values = self.compression_free_conv_stem_input(token_values)

        token_values, num_groups = self._interleave_registers(token_values)
        bsz, seqlen, _ = token_values.shape

        if do_idx is not None: # (CW)
            do_idx_pre_reg = do_idx                             # indices of dropped-out channels without registers interleaved       
            do_idx = (token_values.sum(axis=2)==0).squeeze(0)   # recompute do_idx after interleaving registers


        # Now if using Learable Dropout, replace dropped-out channels with a learned but fixed parameter vector.
        if self.dropout_vec is not None:
            token_values[:,do_idx,:] = self.dropout_vec

        h = self.tok_embeddings(token_values)

        # (CW) - COMBINE SLIDING WINDOW MASK AND DOCUMENT MASK 
        SLIDING_WINDOW = self.sliding_window
        def sliding_window_func(b, h, q_idx, kv_idx):
            # Self-attention case
            return (q_idx - kv_idx).abs() <= SLIDING_WINDOW
        mask_mod_slide = sliding_window_func
        mask = create_document_mask(seq_lens*2, base_mask_mod=mask_mod_slide) # Hardcoding for CR=1 with interleave_registers thing.

        visualize_attention_masks = False
        if visualize_attention_masks:
            from .utils import visualize_attention_mask
            torch._dynamo.config.disable = True
            visualize_attention_mask(mask, title_suffix="encoder")
            torch._dynamo.config.disable = False
            
        if tok_idx is not None:
            tok_idx = tok_idx.repeat_interleave(repeats=2,dim=1)
            tok_idx = tok_idx.squeeze().squeeze() # make it the right size for RoPE.


        if print_layerwise_activation_stats and do_idx is not None: # (CW)
            print(f"{do_idx.sum()=} and {(~do_idx).sum()=}")
            print(f"{token_values.shape=}")


        h, repa_loss = super().forward(h,                   # BaseTransformer.forward
                                       tok_idx=tok_idx, 
                                       mask=mask,
                                       attn_impl=attn_impl, 
                                       repa_target=repa_target, 
                                       num_groups=num_groups, 
                                       original_seqlen=orig_seqlen, 
                                       downsample_factor=self.downsample_factor,
                                       do_idx=do_idx,
        )

        h, non_regs = self._extract_registers_and_non_registers(h, num_groups, original_seqlen=orig_seqlen, return_non_registers=distill_target is not None)


        if print_layerwise_activation_stats and do_idx is not None: # (CW)
            h_normed = self.norm(h) # (CW)
            print(f"\nEncoder output norm (drop-out): mean={h[:, do_idx_pre_reg, :].mean().item():.6f}, std={h[:, do_idx_pre_reg, :].std().item():.6f}", end=" --> ") # (CW)
            print(f"mean={h_normed[:, do_idx_pre_reg, :].mean().item():.6f}, std={h_normed[:, do_idx_pre_reg, :].std().item():.6f}") # (CW)
            
            print(f"Encoder output norm (non-drop): mean={h[:, ~do_idx_pre_reg, :].mean().item():.6f}, std={h[:, ~do_idx_pre_reg, :].std().item():.6f}", end=" --> ") # (CW)
            print(f"mean={h_normed[:, ~do_idx_pre_reg, :].mean().item():.6f}, std={h_normed[:, ~do_idx_pre_reg, :].std().item():.6f}") # (CW)
            logits = self.output(h_normed) # (CW)
        else:
            logits = self.output(self.norm(h)) # (CW) - was this


        logits, losses = self.bottleneck(logits)
        if distill_target is not None:
            losses['encoder_distill'] = (1 - F.cosine_similarity(self.distill_output(self.distill_norm(non_regs)), distill_target, dim=-1).mean()) * 0.1 

        if repa_target is not None:
            losses["encoder_repa_loss"] = repa_loss#.mean()
        return logits, losses

    def bottleneck(self, h,):
        losses = {}
        latent = h
        b, l, d = h.shape
        if "kl" in self.bottleneck_type:
            mean, logvar = h.chunk(2, dim=-1)
            logvar = logvar.clamp(min=-3)
            std = torch.exp(0.5 * logvar)
            latent = mean + std * torch.randn_like(mean)
            kl_div_loss = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
            losses["kl"] = kl_div_loss.mean()

        if "mmd" in self.bottleneck_type and self.training:
            losses["mmd"] = mmd_imq(latent.view(b*l, d).float(), torch.randn((b*l,d), dtype=torch.float32, device=latent.device,), 10.0) 

        if "sim" in self.bottleneck_type:
            latent, codes, simvq_loss = self.quantizer(h)
            losses["simvq_commit_loss"] = simvq_loss

        if "fsq" in self.bottleneck_type:
            latent, codes = self.quantizer(h)

        return latent, losses

    def reset_parameters(self, init_std=None):
        # Either use fixed base std or sqrt model dim
        super().reset_parameters()
        if init_std is None:
            init_std = self.init_base_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )

        nn.init.trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        
        if self.distill:
            self.distill_norm.reset_parameters()
            nn.init.trunc_normal_(
                self.distill_output.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
            nn.init.zeros_(self.distill_output.bias)

        nn.init.trunc_normal_(
            self.registers,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        if self.use_compression_free_conv_stem:
            self.compression_free_conv_stem_input.reset_parameters(init_std)

    def init_weights(self):
        super().init_weights()
        self.reset_parameters()

class EncoderDecoder(nn.Module):
    def __init__(self, args: DecoderTransformerArgs):
        super().__init__()

        self.encoder = EncoderTransformer(args)
        self.decoder = DecoderTransformer(args)
        self.input_output_dim = args.input_dim
        self.encoder_dropout = args.decoder_encoder_dropout
        self.decoder_timestep_dropout = args.decoder_timestep_dropout
        self.global_sigma = args.stft_global_sigma

        self.num_fine_time_pts = args.num_fine_time_pts
        self.dont_noise_chan_xyz = args.dont_noise_chan_xyz
        self.rope_dim = args.rope_dim
        self.tok_idx_type = args.tok_idx_type



    def forward(
            self,
            encoder_input: torch.Tensor,
            decoder_input: torch.Tensor,
            t: torch.Tensor,
            chan_pos: torch.Tensor,
            chan_pos_discrete: torch.Tensor,
            chan_id: torch.Tensor,
            t_coarse: torch.Tensor,
            seq_lens: torch.Tensor,
            target: Optional[torch.Tensor] = None,
            distill_target: Optional[torch.Tensor] = None,
            time_masks: Optional[torch.Tensor] = None,
            channel_loss_weighting: Optional[torch.Tensor] = None, # [1, 1, input_dim*2]
            encoder_repa_target: Optional[torch.Tensor] = None,
            decoder_repa_target: Optional[torch.Tensor] = None,
            freq_masks: Optional[torch.Tensor] = None, # 0s where to not compute loss, 1s where to compute loss [B, 1, C]               
    ):


        if encoder_input.ndim==2:
            encoder_input = encoder_input.unsqueeze(0)
            target = target.unsqueeze(0) # doing to get rid of broadcast warning from DecoderTransformer.compute_losses
            chan_pos = chan_pos.unsqueeze(0)
            chan_pos_discrete = chan_pos_discrete.unsqueeze(0)
            chan_id = chan_id.unsqueeze(0)
            t_coarse = t_coarse.unsqueeze(0)


        ## Options for tok_idx.  Choose 1.
        if self.tok_idx_type is None:
            tok_idx = None          # this will just use args.model.max_seqlen to construct 1D-RoPE (but requires max_seqlen way too long).
        elif self.tok_idx_type == "t_coarse" and self.rope_dim==1:
            tok_idx = t_coarse      # this ignores channel and just uses coarse time in 1D-RoPE
        elif self.tok_idx_type == "chan_id" and self.rope_dim==1:
            tok_idx = chan_id       # this uses channel id in 1D-RoPE                             # this is same as hstack(arange(seq_lens)) below when seq_len = num_chans, ie chop_signals_only
        elif self.tok_idx_type == "stack_arange_seqlen" and self.rope_dim==1:
            tok_idx = torch.hstack([torch.arange(sl) for sl in seq_lens]).unsqueeze(0).unsqueeze(-1) # This has a different tok_id value for each element in sequence (chan or tc).
        elif self.tok_idx_type == "{x,y,z,tc}" and self.rope_dim==4: 
            tok_idx = torch.cat((chan_pos_discrete,t_coarse), dim=2)
        else:
            print(f"Dont understand {self.tok_idx_type=} and {self.rope_dim}")
            die


        do_idx = (encoder_input.sum(axis=2)==0).squeeze(0) # indices of dropped-out channels (CW) 
        # do_idx = None # [Set do_idx to None to disable printing of activation stats comparing channel drop-out]

        enc_out, enc_losses = self.encoder(encoder_input, 
                                           distill_target=distill_target,       # (CW) - None
                                           repa_target=encoder_repa_target,     # (CW) - None
                                           mask=None,
                                           seq_lens=seq_lens,                   # (CW) - for document masking
                                           tok_idx=tok_idx,                    # (CW) - pass in coarse time index for 1D RoPE
                                           do_idx=do_idx,                      # indices of dropped-out channels (CW)
        )

        dec_out, dec_losses = self.decoder(tokens=decoder_input,
                                           cross_attended=enc_out,
                                           timeD=t, 
                                           target=target, 
                                           time_masks=time_masks,                               # (CW) - None
                                           channel_loss_weighting=channel_loss_weighting,       # (CW) - None
                                           repa_target=decoder_repa_target,                     # (CW) - None
                                           freq_masks = freq_masks,                             # (CW) - masks out bad (all-zero) channels [B, 1, C]  
                                           mask=None,
                                           cross_attn_mask=None,
                                           seq_lens=seq_lens,                   # (CW) - for document masking in self-attention
                                           cross_seq_lens=seq_lens,             # (CW) - for document masking in cross-attention (with CR=1)
                                           tok_idx=tok_idx,                     # (CW) - pass in coarse time index for 1D RoPE
                                           cross_tok_idx=tok_idx,               # (CW) - pass in coarse time index for 1D RoPE (with CR=1)
                                           do_idx=do_idx, #.squeeze(0) if do_idx is not None else None,  # indices of dropped-out channels (CW)
        )

        return dec_out, enc_losses, dec_losses


    @torch.no_grad()
    def sample(self, encoder_input: torch.Tensor, seq_lens: torch.Tensor, tok_idx: torch.Tensor, sample_steps: int = 50, cfg: float = 1.0):

        device = encoder_input.device
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        with torch.autocast(device.type, dtype=dtype):

            do_idx = (encoder_input.sum(axis=2)==0).squeeze(0) # indices of dropped-out channels (CW) 

            # do_idx = None # [Set do_idx to None to disable printing of activation stats comparing channel drop-out]
            enc_out, _ = self.encoder(
                            token_values=encoder_input, 
                            seq_lens=seq_lens,
                            tok_idx=tok_idx,
                            do_idx=do_idx,
            )
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

            bsz, seqlen, dim = enc_out.shape
            dt_time = torch.tensor([1.0 / sample_steps] * bsz, device=enc_out.device).view(-1)


            z = self.global_sigma*torch.randn_like(encoder_input).to(enc_out.device) # (CW) - was init to rand
            # z = torch.zeros_like(encoder_input).to(enc_out.device) # (CW) - trying init to zeros

            # Do not noise channel {x,y,z}-position in eeg_signal
            if self.dont_noise_chan_xyz:
                if dim==131 or dim==35:
                    z[:,:,:3] = encoder_input[:,:,:3]
                else:
                    print("NOTE: EEG channel {x,y,z}-position was never concatenated into signal.")
                    import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)


            dt = dt_time.unsqueeze(-1).unsqueeze(-1) # (CW) - added dbl unsqueeze(-1)

            outputs = []
            for i in range(sample_steps, 0, -1):
                t = dt_time * i
                t_model = t.unsqueeze(1).unsqueeze(1)

                vc, _ = self.decoder(tokens=z.unsqueeze(1),
                                     cross_attended=enc_out, 
                                     timeD=t_model, 
                                     seq_lens=seq_lens,                   # (CW) - for document masking in self-attention
                                     cross_seq_lens=seq_lens,             # (CW) - for document masking in cross-attention (with CR=1)
                                     tok_idx=tok_idx,                     # (CW) - pass in coarse time index for 1D RoPE
                                     cross_tok_idx=tok_idx,               # (CW) - pass in coarse time index for 1D RoPE (with CR=1)               
                )

                if cfg != 1.0:
                    # vc_uncond, _ = self.decoder(z, torch.zeros_like(enc_out), t_model) # (CW) - was this
                    vc_uncond, _ = self.decoder(tokens=z.unsqueeze(1),
                                                cross_attended=torch.zeros_like(enc_out), 
                                                timeD=t_model, 
                                                seq_lens=seq_lens,                          # (CW) - for document masking in self-attention
                                                cross_seq_lens=seq_lens,                    # (CW) - for document masking in cross-attention (with CR=1)
                                                tok_idx=tok_idx,                            # (CW) - pass in coarse time index for 1D RoPE
                                                cross_tok_idx=tok_idx,                      # (CW) - pass in coarse time index for 1D RoPE (with CR=1)                
                    )

                    vc = vc_uncond + cfg * (vc - vc_uncond) # starts at unconditioned, moves toward conditioned as cfg increases
                    
                z = z - dt * vc # <-- (CW) - should this be t or t_model or dt?

                # Do not noise channel {x,y,z}-position in eeg_signal
                if self.dont_noise_chan_xyz:
                    if dim==131 or dim==35:
                        z[:,:,:3] = encoder_input[:,:,:3]
                    else:
                        print("NOTE: EEG channel {x,y,z}-position was never concatenated into signal.")
                        import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

                outputs.append(z)
            
            return z, outputs



    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def init_weights(self):
        self.encoder.init_weights()
        self.decoder.init_weights()




# Optional policy for activation checkpointing. With None, we stick to the default (defined distributed.py: default_no_recompute_ops)
def get_no_recompute_ops():
    return None


# Optional and only used for fully shard options (fsdp) is choose. Highly recommanded for large models
def build_fsdp_grouping_plan(model_args: DecoderTransformerArgs) -> List[Tuple[str, bool]]:
    group_plan: List[Tuple[str, bool]] = []

    # # 1. Encoder Input
    group_plan.append(("encoder.output", False)) # <-- Changed to True
    group_plan.append(("decoder.output", False)) # Final output for main loss
    if model_args.decoder_repa_index != inf:
        group_plan.append(("decoder.repa_proj", False))
    if model_args.encoder_repa_index != inf:
        group_plan.append(("encoder.repa_proj", False))

    # 2. Encoder Transformer Blocks
    for i in range(model_args.n_layers):
        group_plan.append((f"encoder.layers.{i}", False))

    # 3. Decoder Transformer Blocks
    for i in range(model_args.n_layers):
        group_plan.append((f"decoder.layers.{i}", False))

    # 4. Add Decoder and Encoder themselves
    group_plan.append(("encoder", False))
    group_plan.append(("decoder", False))

    return group_plan