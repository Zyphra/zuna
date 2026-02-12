from lingua.transformer import *


def apply_rotary_emb_xattn(
    xq: torch.Tensor,
    xk: torch.Tensor,
    seq_dim: int,
    freqs_cis_q: torch.Tensor,
    freqs_cis_k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    freqs_cis_q = reshape_for_broadcast(
        freqs_cis_q, xq_, seq_dim
    ).float()  # S D/2 2 2 -> 1 S 1 D/2 2 2
    freqs_cis_k = reshape_for_broadcast(
        freqs_cis_k, xk_, seq_dim
    ).float()  # S D/2 2 2 -> 1 S 1 D/2 2 2
    xq_out = (xq_ * freqs_cis_q).sum(5).flatten(3)
    xk_out = (xk_ * freqs_cis_k).sum(5).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class AdaRMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, emb_dim, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Linear(emb_dim, dim, bias=True)

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        x = probe.log_stats(x, "resid")
        output = self._norm(x.float())
        return (output * self.weight(c).float()).type_as(x)

    def reset_parameters(self):
        # bias to ones, weight to 0s
        nn.init.ones_(self.weight.bias)
        nn.init.zeros_(self.weight.weight)


class CrossAttention(nn.Module):
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
        xq: torch.Tensor,
        xkv: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        cross_tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        # B S D
        assert attn_impl == "flex_attention", "Only flex_attention is supported for now"
        bsz, seq_len_q, dim = xq.shape
        _, seq_len_kv, _ = xkv.shape
        xq = self.wq(xq.view_as(xq))
        xk = self.wk(xkv.view_as(xkv))
        xv = self.wv(xkv.view_as(xkv))

        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len_q, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len_kv, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len_kv, self.n_kv_heads, self.head_dim)

        if self.rope_dim==0:
            pass
        elif self.rope_dim==1:
            if tok_idx is not None and cross_tok_idx is not None:
                xq, xk = apply_rotary_emb_xattn(
                    xq, xk, 1, freq_cis[tok_idx], freq_cis[cross_tok_idx]
                )
            else:
                xq, xk = apply_rotary_emb_xattn(
                    xq, xk, 1, freq_cis[0:seq_len_q], freq_cis[0:seq_len_kv]
                )
        elif self.rope_dim==4:

            # Build freqcis_4RoPE by indexing freq_cis with each dimension of tok_idx separately and concatenating
            # Cat along a new dimension to get [S, head_dim//2, 2, 2]
            freqcis_parts = []
            freqcis_cross_parts = []
            for i in range(self.rope_dim):
                freqcis_parts.append(freq_cis[tok_idx[:, i]])
                freqcis_cross_parts.append(freq_cis[cross_tok_idx[:, i]])
            freqcis_4RoPE = torch.cat(freqcis_parts, dim=1)
            freqcis_cross_4RoPE = torch.cat(freqcis_cross_parts, dim=1)


            xq, xk = apply_rotary_emb_xattn(
                xq, xk, 1, freqcis_4RoPE, freqcis_cross_4RoPE
            )

        else:
            print(f"I dont know how to handle {self.rope_dim=} inside xattn.CrossAttention.forward")
            import IPython; print('\n\nDebug:'); IPython.embed(); import time;  time.sleep(0.3)

        # This condition helps us be easily compatible
        # with inference by adding a pluggable KVCache
        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        if attn_impl == "flex_attention":
            assert mask is None or isinstance(mask, BlockMask)
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            # print(f"Inside xattn.CrossAttention.forward, xq.device.type={xq.device.type}")
            # Use uncompiled flex_attention on CPU; flex_attention_comp triggers device-mismatch with block masks on CPU
            if xq.device.type == "cuda":
                output = flex_attention_comp(xq, xk, xv, block_mask=mask)
            else:
                output = flex_attention(xq, xk, xv, block_mask=mask)
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D


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


class FourierConditioner(nn.Module):
    def __init__(
        self,
        output_dim: int,
        input_dim: int = 1,
        std: float = 0.02,
        min_val: float = 0.0,
        max_val: float = 1.0,
    ):
        super().__init__()
        assert input_dim == 1
        assert output_dim % 2 == 0
        self.output_dim = output_dim
        self.register_buffer("weight", torch.randn([output_dim // 2, input_dim]) * std)
        self.min_val, self.max_val = min_val, max_val
        self.proj = nn.Linear(output_dim, output_dim)

    def forward(self, x: list[float], device=None):
        x = (x - self.min_val) / (self.max_val - self.min_val)
        f = (2 * torch.pi * x.float() @ self.weight.T).type_as(x)
        return self.proj(torch.cat([f.cos(), f.sin()], dim=-1))

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.output_dim ** (-0.5))

        self.register_buffer("weight", torch.randn([self.output_dim // 2, 1]).to(self.proj.weight.device) * init_std)

        nn.init.trunc_normal_(
            self.proj.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )
        nn.init.zeros_(self.proj.bias)


@dataclass
class DecoderArgs(BaseTransformerArgs):

    t_dim: int = 64
    n_heads: int = 8
    seqlen_t: bool = False


class DecoderBlock(nn.Module):
    def __init__(self, args: DecoderArgs):
        super().__init__()

        assert (args.head_dim is not None) or (args.n_heads is not None), (
            "Should specify at least head_dim or n_heads"
        )
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // args.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        assert args.n_heads % self.n_kv_heads == 0
        assert args.dim % args.n_heads == 0

        self.cross_attention = CrossAttention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
            rope_dim=args.rope_dim,
        )
        self.cross_attention_x_norm = AdaRMSNorm(
            args.t_dim, args.dim, eps=args.norm_eps
        )

        self.seqlen_t = args.seqlen_t
        if args.seqlen_t:
            self.cross_attention_y_norm = RMSNorm(
                args.dim, eps=args.norm_eps
            )
        else:
            self.cross_attention_y_norm = AdaRMSNorm(
                args.t_dim, args.dim, eps=args.norm_eps
            )

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
        self.attention_norm = AdaRMSNorm(args.t_dim, args.dim, eps=args.norm_eps)
        self.ffn_norm = AdaRMSNorm(args.t_dim, args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        c: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        cross_tok_idx: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[Union[BlockMask, str]] = None,
        cross_attn_mask: Optional[Union[BlockMask, str]] = None,
        attn_impl: str = "sdpa",
        do_idx: Optional[torch.Tensor] = None,
        print_layerwise_activation_stats: bool = False,
    ) -> torch.Tensor:
        
        if print_layerwise_activation_stats and do_idx is not None:

            x_normed = self.cross_attention_x_norm(x, c)
            y_normed = self.cross_attention_y_norm(y, c) if not self.seqlen_t else self.cross_attention_y_norm(y) 

            print(f"\n\tDecoder cross_attn_x_norm: (drop-out) mean={x[:, do_idx, :].mean().item():.6f}, std={x[:, do_idx, :].std().item():.6f}", end=" --> ")
            print(f"mean={x_normed[:, do_idx, :].mean().item():.6f}, std={x_normed[:, do_idx, :].std().item():.6f}") 
                        
            print(f"\tDecoder cross_attn_x_norm: (non-drop) mean={x[:, ~do_idx, :].mean().item():.6f}, std={x[:, ~do_idx, :].std().item():.6f}", end=" --> ")
            print(f"mean={x_normed[:, ~do_idx, :].mean().item():.6f}, std={x_normed[:, ~do_idx, :].std().item():.6f}") 

            print(f"\n\tDecoder cross_attn_y_norm: (drop-out) mean={y[:, do_idx, :].mean().item():.6f}, std={y[:, do_idx, :].std().item():.6f}", end=" --> ") 
            print(f"mean={y_normed[:, do_idx, :].mean().item():.6f}, std={y_normed[:, do_idx, :].std().item():.6f}")

            print(f"\tDecoder cross_attn_y_norm: (non-drop) mean={y[:, ~do_idx, :].mean().item():.6f}, std={y[:, ~do_idx, :].std().item():.6f}", end=" --> ")
            print(f"mean={y_normed[:, ~do_idx, :].mean().item():.6f}, std={y_normed[:, ~do_idx, :].std().item():.6f}") 
        
            x = x + self.cross_attention( 
                x_normed, 
                y_normed, 
                freq_cis,
                tok_idx=tok_idx,
                cross_tok_idx=cross_tok_idx,
                mask=cross_attn_mask,
                attn_impl=attn_impl,
            )

        else:
            x = x + self.cross_attention( 
                self.cross_attention_x_norm(x, c),
                self.cross_attention_y_norm(y, c) if not self.seqlen_t else self.cross_attention_y_norm(y),
                freq_cis,
                tok_idx=tok_idx,
                cross_tok_idx=cross_tok_idx,
                mask=cross_attn_mask,
                attn_impl=attn_impl,
            )



        if print_layerwise_activation_stats and do_idx is not None:

            x_normed = self.attention_norm(x, c) 

            print(f"\n\tDecoder self attn_norm: (drop-out) mean={x[:, do_idx, :].mean().item():.6f}, std={x[:, do_idx, :].std().item():.6f}", end=" --> ") 
            print(f" mean={x_normed[:, do_idx, :].mean().item():.6f}, std={x_normed[:, do_idx, :].std().item():.6f}") 
            
            print(f"\tDecoder self attn_norm: (non-drop) mean={x[:, ~do_idx, :].mean().item():.6f}, std={x[:, ~do_idx, :].std().item():.6f}", end=" --> ") 
            print(f"mean={x_normed[:, ~do_idx, :].mean().item():.6f}, std={x_normed[:, ~do_idx, :].std().item():.6f}") 
        
            h = x + self.attention( 
                x_normed, 
                freq_cis,
                tok_idx=tok_idx,
                mask=self_attn_mask,
                attn_impl=attn_impl,
            )

            h_normed = self.ffn_norm(h, c) 

            print(f"\n\tDecoder ffn_norm: (drop-out) mean={h[:, do_idx, :].mean().item():.6f}, std={h[:, do_idx, :].std().item():.6f}", end=" --> ") 
            print(f"mean={h_normed[:, do_idx, :].mean().item():.6f}, std={h_normed[:, do_idx, :].std().item():.6f}") 
            
            print(f"\tDecoder ffn_norm: (non-drop) mean={h[:, ~do_idx, :].mean().item():.6f}, std={h[:, ~do_idx, :].std().item():.6f}", end=" --> ") 
            print(f"mean={h_normed[:, ~do_idx, :].mean().item():.6f}, std={h_normed[:, ~do_idx, :].std().item():.6f}") 

            out = h + self.feed_forward(h_normed) 

        else:

            h = x + self.attention(
                self.attention_norm(x, c),
                freq_cis,
                tok_idx=tok_idx,
                mask=self_attn_mask,
                attn_impl=attn_impl,
            )
            out = h + self.feed_forward(self.ffn_norm(h, c))

        return out

    def init_weights(self, init_std=None, factor=1.0):
        self.cross_attention.reset_parameters(init_std, factor)
        self.cross_attention_x_norm.reset_parameters()
        self.cross_attention_y_norm.reset_parameters()

        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()

        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()
