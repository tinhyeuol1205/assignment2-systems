import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    max_logits = torch.max(logits, dim = -1, keepdim=True).values
    shifted_logits = logits - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim = -1, keepdim=True))
    target_loss = -torch.gather(shifted_logits, dim = -1, index = targets.unsqueeze(-1).to(dtype=torch.int64))
    loss = log_sum_exp + target_loss

    return loss.mean()
