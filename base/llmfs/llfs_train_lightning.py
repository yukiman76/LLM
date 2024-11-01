import os
import torch
os.environ['HF_HOME'] = '/mnt/data/hf_cache/'
os.environ['HF_DATASETS_CACHE'] = '/mnt/data/datasets/'
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

import mlflow
import tiktoken
from torch import nn
from torch.optim import Adam
from datetime import datetime
from importlib.metadata import version
import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import MLFlowLogger
from llfs_data import create_dataloader
from llfs_model import LlamaModel2
from llfs_config import get_config

pkgs = ["matplotlib", "torch", "tiktoken"]
for p in pkgs:
    print(f"{p} version: {version(p)}")

torch.manual_seed(129)

# Lightning Module for Training
class LlamaLightningModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = LlamaModel2(
            vocab_size=config['vocab_size'],
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
    tokenizer.pad_token = tokenizer.eot_token
    tokenizer.padding_side = "right"  
    tokenizer.eos_token_id = tokenizer.eot_token 

    config["vocab_size"] = tokenizer.n_vocab
    config["learning_rate"] = 1e-3
    config["epochs"] = 1

    # Data loader preparation
    data_loader = create_dataloader(
        sdir='data', 
        tokenizer=tokenizer,
        batch_size=config['batch_size'],
        max_length=config['max_seq_length'],
        stride=config['max_seq_length'],
        num_workers=int(os.cpu_count() /1.5) # we want to use most of the CPU's to handle data loading
    )

    
    # Model initialization
    model = LlamaLightningModule(config=config)

    # Set up MLflow tracking
    sDate = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    sModelPath = f"llfs_train_{sDate}"
    # Create out logger
    mlf_logger = MLFlowLogger(experiment_name=sModelPath, tracking_uri="file:./mlruns")
    # mlflow.start_run(run_name=sModelPath)
    # mlflow.log_params(config)

    # Trainer for multi-GPU support
    trainer = L.Trainer(
        devices=torch.cuda.device_count(),
        accelerator="gpu",
        max_epochs=config['epochs'],
        strategy=DDPStrategy(find_unused_parameters=False),
        log_every_n_steps=10,
        logger=mlf_logger
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
