import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional
import math

class CausalConv2DStem(nn.Module):
    """
    A 'beefier' Causal 2D Convolutional Stem using standard convolutions.

    Processes input [B, T, C=2*F] -> [B, T, F_out]. Uses standard Conv2D layers
    for increased parameters. Maintains strict time causality via manual padding.
    Uses PyTorch's truncated normal initialization.
    Structure: Pointwise -> Act -> Conv2D -> Act -> Conv2D -> Act -> Pointwise

    Args:
        input_features (int): Input features C (even, = 2*F).
        hidden_channels (int): Intermediate convolution channels.
        time_kernel_size (int): Kernel size for time dim (>= 1).
        freq_kernel_size (int): Kernel size for freq dim (>= 1, typically odd).
        compress_channels (bool): If True, output F_out = F. If False, F_out = 2*F.
        activation (nn.Module): Activation function. Default: nn.GELU().
        init_std (float): Std dev for truncated normal weight init. Default: 0.02.
    """
    def __init__(
        self,
        input_features: int,
        hidden_channels: int,
        time_kernel_size: int = 3,
        freq_kernel_size: int = 3,
        compress_channels: bool = True,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        if not (isinstance(input_features, int) and input_features > 0 and input_features % 2 == 0):
            raise ValueError("input_features must be an even positive integer.")
        if not (isinstance(hidden_channels, int) and hidden_channels > 0):
            raise ValueError("hidden_channels must be a positive integer.")
        if not (isinstance(time_kernel_size, int) and time_kernel_size >= 1):
            raise ValueError("time_kernel_size must be >= 1.")
        if not (isinstance(freq_kernel_size, int) and freq_kernel_size >= 1):
             raise ValueError("freq_kernel_size must be >= 1.")

        self.input_features = input_features
        self.freq_dim = input_features // 2
        self.target_channels = 1 if compress_channels else 2
        self._activation = activation() if activation is not None else nn.GELU()

        self._time_pad_left = time_kernel_size - 1
        self._freq_pad_sym = (freq_kernel_size - 1) // 2
        self._causal_padding = (self._freq_pad_sym, self._freq_pad_sym, self._time_pad_left, 0)

        self._pointwise1 = nn.Conv2d(2, hidden_channels, 1, 1, bias=False)
        self._conv1 = nn.Conv2d(hidden_channels, hidden_channels, (time_kernel_size, freq_kernel_size), 1, padding=0, bias=False)
        self._conv2 = nn.Conv2d(hidden_channels, hidden_channels, (time_kernel_size, freq_kernel_size), 1, padding=0, bias=False)
        self._pointwise2 = nn.Conv2d(hidden_channels, self.target_channels, 1, 1, bias=True)
        self.output_features = self.target_channels * self.freq_dim


    def reset_parameters(self, std: float):
        """Initialize Conv2d weights with truncated normal and biases with zeros."""
        def init_the_shit(module):
            if isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3*std, b=3*std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(lambda module: init_the_shit(module))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Args: x [B, T, C=2*F]. Returns: [B, T, F_out]. """
        B, T, C = x.shape
        if C != self.input_features: raise ValueError(f"Input C={C} != expected {self.input_features}")

        x = rearrange(x, 'b t (c f) -> b c t f', c=2, f=self.freq_dim)
        x = self._activation(self._pointwise1(x))
        x = F.pad(x, self._causal_padding)
        x = self._activation(self._conv1(x))
        x = F.pad(x, self._causal_padding)
        x = self._activation(self._conv2(x))
        x = self._pointwise2(x) # [B, target_channels, T, F]
        x = rearrange(x, 'b c t f -> b t (c f)') # [B, T, F_out]
        return x

    def get_output_dim(self) -> int:
        return self.output_features