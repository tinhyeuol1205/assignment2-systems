import torch
from cs336_basics_new.nn import ModelConfig, TransformerLM
from cs336_basics_new.training import TrainingConfig, Trainer
from cs336_basics_new.data import LMDataset

def main(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    train_data_path: str,
    val_data_path: str,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = LMDataset(train_data_path, model_config.max_seq_len)
    val_dataset = LMDataset(val_data_path, model_config.max_seq_len)

    model = TransformerLM(model_config).to(device)

    trainer = Trainer(
        model, 
        training_config,
        train_dataset, 
        val_dataset
    )
    trainer.train()

if __name__ == "__main__":
    model_config = ModelConfig(
        vocab_size=10000, 
        d_model=512, 
        d_ff=1344, 
        theta=10000.0, 
        num_heads=16, 
        num_layers=4, 
        max_seq_len=256
    )

    training_config = TrainingConfig(
        batch_size=32,
        min_lr=6e-5,
        max_lr=6e-4,
        weight_decay=0.01,
        betas=(0.9, 0.98),
        eps=1e-8,
        max_l2_norm=1.0,
        num_epochs=3,
        warmup_iters=3000,
        cosin_cycle_iters=40000,
        eval_interval=1000,
        checkpoint_dir="../checkpoint",
        wandb_project="gpt-from-scratch",
        wandb_run_name="gpt-nano"
    )
    main(model_config, training_config, "../data/train.bin", "../data/val.bin")