import math
from typing import Iterable
import torch

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    global_l2_norm = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        else:
            global_l2_norm += (p.grad ** 2).sum().item()
    global_l2_norm = math.sqrt(global_l2_norm)
    if global_l2_norm > max_l2_norm and global_l2_norm != 0:
        for p in parameters:
            if p.grad is None:
                continue
            else:
                p.grad *= max_l2_norm / global_l2_norm
