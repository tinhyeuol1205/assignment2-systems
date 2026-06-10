import math
from typing import Callable, Optional

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas: tuple[float, float], eps, weight_decay):
        if lr < 0:
            raise ValueError("Learning rate must be non-negative")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 1)
                m = torch.zeros_like(p.data) if state.get("m") is None else state.get("m")
                v = torch.zeros_like(p.data) if state.get("v") is None else state.get("v")
                g = p.grad.data
                p.data -= lr * weight_decay * p.data
                alpha_t = lr * math.sqrt(1 - betas[1]**t) / (1 - betas[0]**t)
                m = betas[0] * m + (1 - betas[0]) * g
                v = betas[1] * v + (1 - betas[1]) * g**2
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        return loss