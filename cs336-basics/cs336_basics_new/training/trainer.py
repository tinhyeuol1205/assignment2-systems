from cs336_basics_new.nn import TransformerLM
from cs336_basics_new.optim import get_lr_cosin_schedule
from cs336_basics_new.nn import cross_entropy
from cs336_basics_new.nn import gradient_clipping
from cs336_basics_new.optim import AdamW
from torch.utils.data import DataLoader
from .config import TrainingConfig
import torch
import typing
import os
import wandb

class Trainer:
    def __init__(self, model: TransformerLM, config: TrainingConfig, train_dataset, val_dataset):
        self.model = model
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.max_lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps
        )
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = next(model.parameters()).device
        self.wandb_run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                "learning_rate": config.max_lr,
                "architecture": "TransformerLM",
                "num_heads": model.config.num_heads,
                "num_layers": model.config.num_layers,
                "d_model": model.config.d_model,
                "d_ff": model.config.d_ff,
                "max_seq_len": model.config.max_seq_len,
                "theta": model.config.theta,
                "weight_decay": config.weight_decay,
                "betas": config.betas,
                "eps": config.eps,
                "max_l2_norm": config.max_l2_norm,
            }
        )


    def train(self, resume_from: str = None):
        global_step = 0
        if resume_from: global_step = self.load_checkpoint(resume_from)
        train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=True)
        for epoch in range(self.config.num_epochs):
            for batch in train_loader:
                x_train, y_train = batch
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)
                if global_step % self.config.eval_interval == 0:
                    val_loss = self.evaluate(val_loader)
                    self.wandb_run.log({"val_loss": val_loss}, step=global_step)
                    print(f"Epoch {epoch}, Iteration {global_step}, Val Loss {val_loss}")
                    self.save_checkpoint(global_step, os.path.join(self.config.checkpoint_dir, f"checkpoint_{global_step}.pt"))
                self.model.train()
                lr = get_lr_cosin_schedule(global_step, self.config.max_lr, self.config.min_lr, self.config.warmup_iters, self.config.cosin_cycle_iters)
                for group in self.optimizer.param_groups:
                    group['lr'] = lr
                loss = self._training_step(x_train, y_train)
                self.wandb_run.log({
                    "train_loss": loss, 
                    "learning_rate": lr,
                    "iteration": global_step
                }, step=global_step)
                print(f"Iteration {global_step}, Loss {loss}, Learning Rate {lr}")
                global_step += 1
            print(f"Epoch {epoch}, Loss {loss}")
        self.wandb_run.finish()

    def evaluate(self, val_loader: DataLoader):
        self.model.eval()
        with torch.no_grad():
            val_losses = torch.zeros(len(val_loader))
            for i, (x, y) in enumerate(val_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                predicted_y = self.model(x)
                val_losses[i] = cross_entropy(predicted_y, y)
        val_loss = val_losses.mean().item()
        return val_loss
    def _training_step(self, x: torch.Tensor, y: torch.Tensor):
        predicted_y = self.model(x)
        loss = cross_entropy(predicted_y, y)
        self.optimizer.zero_grad()
        loss.backward()
        gradient_clipping(self.model.parameters(), self.config.max_l2_norm)
        self.optimizer.step()
        return loss.item()


    def save_checkpoint(self, iteration: int, checkpoint_path: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint = {
            "iteration": iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["iteration"]