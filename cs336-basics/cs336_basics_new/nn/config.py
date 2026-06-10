from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    d_ff: int
    theta: float
    num_heads: int
    num_layers: int
    max_seq_len: int