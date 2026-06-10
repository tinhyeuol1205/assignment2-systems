import math

def get_lr_cosin_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosin_cycle_iters: int):
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    elif it < cosin_cycle_iters:
        return min_learning_rate + 1/2 * (max_learning_rate-min_learning_rate) * (1 + math.cos(math.pi * (it - warmup_iters) / (cosin_cycle_iters - warmup_iters)))
    else:
        return min_learning_rate