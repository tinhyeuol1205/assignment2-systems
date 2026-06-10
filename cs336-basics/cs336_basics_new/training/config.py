from dataclasses import dataclass

@dataclass
class TrainingConfig:
    max_lr: float
    min_lr: float
    warmup_iters: int
    cosin_cycle_iters: int
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    batch_size: int
    num_epochs: int
    eval_interval: int
    max_l2_norm: float
    checkpoint_dir: str
    wandb_project: str
    wandb_run_name: str | None = None