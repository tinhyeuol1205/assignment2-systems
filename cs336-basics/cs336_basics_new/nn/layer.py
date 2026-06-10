import torch
from torch import nn
from einops import einsum, rearrange
import math

class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device=None, dtype=None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty((d_out, d_in), device=device, dtype=dtype))
        sigma = math.sqrt(2.0 / (d_in + d_out))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty((vocab_size, d_model), device=device, dtype=dtype))
        sigma = 1.0
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3.0, b=3.0)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        rmsnorm = (x / rms) * self.weight
        return rmsnorm.to(in_dtype)