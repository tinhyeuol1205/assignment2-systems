import torch
import time

from cs336_basics_new.optim import AdamW
from cs336_basics_new.nn import cross_entropy, TransformerLM, ModelConfig
import torch.cuda.nvtx as nvtx

model_config = ModelConfig(
    vocab_size=10000,
    d_model=1024,
    d_ff=4096,
    num_heads=16,
    num_layers=24,
    max_seq_len=512,
    theta=10_000
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TransformerLM(model_config).to(device)
optimizer = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

batch_size = 4
max_seq_len = 512
sample_input = torch.randint(0, model_config.vocab_size, (batch_size, max_seq_len), device=device)

model.eval()
with torch.no_grad():
    output = model(sample_input)
    torch.cuda.synchronize()

model.train()
forward_time = []
backward_time = []
optimizer_time = []
for i in range(3):
    start = time.time()
    nvtx.range_push("Forward")
    output = model(sample_input)
    torch.cuda.synchronize() # Wait for GPU to finish
    nvtx.range_pop()
    forward_stone = time.time()
    forward_time.append(forward_stone - start)
    start_backward = time.time()
    nvtx.range_push("Backward")
    simulate_loss = cross_entropy(output, sample_input)
    optimizer.zero_grad()
    simulate_loss.backward()
    torch.cuda.synchronize() # Wait for GPU to finish
    nvtx.range_pop()
    backward_stone = time.time()
    backward_time.append(backward_stone - start_backward) 
    optimizer_start = time.time()
    nvtx.range_push("Optimization")
    optimizer.step()
    torch.cuda.synchronize() # Wait for GPU to finish
    nvtx.range_pop()
    optimizer_stone = time.time()
    optimizer_time.append(optimizer_stone - optimizer_start)
print(f"Average forward time: {sum(forward_time)/len(forward_time):.4f} seconds")
print(f"Average backward time: {sum(backward_time)/len(backward_time):.4f} seconds")
print(f"Average optimization time: {sum(optimizer_time)/len(optimizer_time):.4f} seconds")
    