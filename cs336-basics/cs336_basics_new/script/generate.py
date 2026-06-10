from cs336_basics_new.tokenizer import Tokenizer
from cs336_basics_new.nn import TransformerLM, ModelConfig
import torch 

tokenizer = Tokenizer.from_files("vocab.json", "merges.json")
eos_id = tokenizer.inverse_vocab['<|endoftext|'.encode('utf-8')]
input_ids = tokenizer.encode("Hello, my name is")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_config = ModelConfig(
    vocab_size=10000,
    d_model=512,
    d_ff=1344,
    theta=10000.0,
    num_heads=16,
    num_layers=4,
    max_seq_len=1024
)
model = TransformerLM(model_config).to(device)

model.load_state_dict(torch.load("model.pth").get("model_state_dict", model.state_dict()))

input_ids = torch.tensor([input_ids], device=device)

output_ids = model.generate(input_ids, max_new_tokens=100, temperature=0.8, top_p=0.9, eos_token_id=eos_id)

print(tokenizer.decode(output_ids[0].tolist()))
