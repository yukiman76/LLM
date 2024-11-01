import os
import torch
import mlflow
import tiktoken
from torch import nn
from torch.optim import Adam
from datetime import datetime
from importlib.metadata import version
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
from llfs_data import create_dataloader
from llfs_model import LlamaModel2
from llfs_config import get_config

# Configurations for local disk mount and caching
LOCAL_DISK_MOUNT = '/mnt/data'
if os.path.exists(LOCAL_DISK_MOUNT):
    os.environ['HF_HOME'] = f'{LOCAL_DISK_MOUNT}/hf_cache/'
    os.environ['HF_DATASETS_CACHE'] = f'{LOCAL_DISK_MOUNT}/datasets/'

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

pkgs = ["matplotlib", "torch", "tiktoken"]
for p in pkgs:
    print(f"{p} version: {version(p)}")

torch.manual_seed(129)

# Lightning Module for Training
class LlamaLightningModule(pl.LightningModule):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.model = LlamaModel2(
            vocab_size,
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.config = config
        self.save_hyperparameters(config)

    def forward(self, x):
        return self.model(x)

    def compute_accuracy(self, preds, labels):
        _, predicted = torch.max(preds, dim=-1)
        correct = (predicted == labels).float()
        accuracy = correct.sum() / len(labels)
        return accuracy

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        output = output.view(-1, self.hparams.vocab_size)
        target = target.view(-1)
        loss = self.loss_fn(output, target)
        accuracy = self.compute_accuracy(output, target)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)


# Main function to prepare data, model, and run training
def main():
    # Load configuration and set parameters
    config = get_config()
    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab
    config["vocab_size"] = vocab_size
    config["learning_rate"] = 1e-3
    config["batch_size"] = 32
    config["epochs"] = 1

    # Data loader preparation
    data_loader = create_dataloader(
        'data', tokenizer,
        batch_size=config['batch_size'],
        max_length=config['max_seq_length'],
        stride=config['max_seq_length'],
        num_workers=10
    )

    # Model initialization
    model = LlamaLightningModule(config=config, vocab_size=vocab_size)

    # Set up MLflow tracking
    sDate = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    sModelPath = f"llfs_train_{sDate}"
    mlflow.start_run(run_name=sModelPath)
    mlflow.log_params(config)

    # Trainer for multi-GPU support
    trainer = Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=config['epochs'],
        strategy=DDPPlugin(find_unused_parameters=False),
        log_every_n_steps=10
    )

    # Training
    trainer.fit(model, data_loader)

    # Save model weights and log with MLflow
    model_path = f"./llmfs_weights_{sDate}.pth"
    torch.save(model.state_dict(), model_path)
    mlflow.pytorch.log_model(model, "model")
    mlflow.end_run()

if __name__ == "__main__":
    main()
