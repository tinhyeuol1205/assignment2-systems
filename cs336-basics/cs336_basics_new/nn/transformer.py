import torch
from torch import nn
from einops import einsum, rearrange
import math
from .layer import Linear, RMSNorm, Embedding
from .config import ModelConfig

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int,):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2((self.w1(x) * torch.sigmoid(self.w1(x))) * self.w3(x))
        
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        sin_cache = []
        cos_cache = []
        for i in range(max_seq_len):
            sin_cache.append([math.sin(i / (theta**((2*k)/d_k))) for k in range(0, d_k//2)])
            cos_cache.append([math.cos(i / (theta**((2*k)/d_k))) for k in range(0, d_k//2)])
        self.register_buffer("sin_cache", torch.tensor(sin_cache, device=device), persistent=False)
        self.register_buffer("cos_cache", torch.tensor(cos_cache, device=device), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        sin = rearrange(self.sin_cache[token_positions], "... d -> ... d 1")
        cos = rearrange(self.cos_cache[token_positions], "... d -> ... d 1")
        x_pair = rearrange(x, "... (d two) -> ... d two", two = 2)
        x_1 = x_pair[..., 0]
        x_2 = x_pair[..., 1]
        x_rotate = rearrange([-x_2, x_1], "two ... d -> ... d two")
        output = x_pair * cos + x_rotate * sin
        return rearrange(output, "... d two -> ... (d two)")

def softmax(x: torch.Tensor, i: int)->torch.Tensor:
    max_v = torch.max(x, dim=i, keepdim=True).values
    return torch.exp(x-max_v)/torch.sum(torch.exp(x-max_v), dim=i, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    score = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / math.sqrt(Q.shape[-1])
    if mask is not None:
        score = score.masked_fill(~mask, float('-inf'))
    score = softmax(score, -1)
    return einsum(score, V, "... q k, ... k d_v -> ... q d_v")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.d_k, max_seq_len)
        else:
            self.rope = None
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        q_heads = rearrange(Q, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads = self.num_heads)
        k_heads = rearrange(K, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads = self.num_heads)
        v_heads = rearrange(V, "... seq_len (heads d_v) -> ... heads seq_len d_v", heads = self.num_heads)

        if self.rope is not None:
            q_heads = self.rope(q_heads, token_positions)
            k_heads = self.rope(k_heads, token_positions)
        seq_len = x.shape[-2]
        mask_tensor = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
        score_heads = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask_tensor)
        score = rearrange(score_heads, "... heads seq_len d_v -> ... seq_len (heads d_v)", heads = self.num_heads)
        
        return self.output_proj(score)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int | None = None, theta: float | None = None):
        super().__init__()
        if theta is not None and max_seq_len is not None:
            self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta)
        else:
            self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        y = x + self.attn(self.ln1(x), token_positions)
        return y + self.ffn(self.ln2(y))

class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = Embedding(config.vocab_size, config.d_model)
        self.layer = nn.ModuleList([TransformerBlock(config.d_model, config.num_heads, config.d_ff, config.max_seq_len, config.theta) for _ in range(config.num_layers)])
        self.ln_final = RMSNorm(config.d_model)
        self.lm_head = Linear(config.d_model, config.vocab_size)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = self.token_embeddings(x)
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        for block in self.layer:
            x = block(x, token_positions)
        x = self.ln_final(x)
        return self.lm_head(x)
    def generate(self, x: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_p: float | None = None, eos_token_id: int | None = None):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self(x[:, -self.config.max_seq_len:])
                if temperature == 0:
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                else:
                    logits = logits[:, -1, :] / temperature
                    if top_p is not None:
                        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        mask = cumulative_probs > top_p
                        mask[:, 1:] = mask[:, :-1].clone()
                        mask[:, 0] = False
                        sorted_logits[mask] = -float('inf')
                        probs = torch.softmax(sorted_logits, dim=-1)
                        next_token_sorted = torch.multinomial(probs, num_samples=1)
                        next_token = torch.gather(sorted_indices, dim=-1, index=next_token_sorted)
                    else:
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, next_token], dim=-1)
                if eos_token_id is not None and next_token == eos_token_id:
                    break
        return x