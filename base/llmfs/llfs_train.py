#!/usr/bin/env python
# coding: utf-8
import os
LOCAL_DISK_MOUNT = '/mnt/data'
if os.path.exists(LOCAL_DISK_MOUNT):
    os.environ['HF_HOME'] = f'{LOCAL_DISK_MOUNT}/hf_cache/'
    os.environ['HF_DATASETS_CACHE'] = f'{LOCAL_DISK_MOUNT}/datasets/'

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
import mlflow
import torch
import tiktoken
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from datetime import datetime
from importlib.metadata import version
# our tools
from llfs_data import create_dataloader 
from llfs_infrence import generate_text, count_parameters
from llfs_model import LlamaModel2
from llfs_config import get_config

pkgs = ["matplotlib",
        "torch",
        "tiktoken"
       ]
for p in pkgs:
    print(f"{p} version: {version(p)}")

torch.manual_seed(129)
    
device: str = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"setting device to {device}\n")


# Function to compute accuracy
def compute_accuracy(preds, labels):
    # Get the index of the max log-probability
    _, predicted = torch.max(preds, dim=-1)
    correct = (predicted == labels).float()
    accuracy = correct.sum() / len(labels)
    return accuracy.item()


def train(epochs=1):
    loss_fn = nn.CrossEntropyLoss()
    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab # Use tokenizer's vocab size
    config = get_config()

    config['epochs'] = epochs
    config["learning_rate"] = 1e-3
    config["metric_function"] = "compute_accuracy"
    config["optimizer"] = "Adam"
    config["loss_function"] = loss_fn.__class__.__name__
    config["tokenizer"] = "cl100k_base" 

    print(f"\nConfig:\n{config}")
    sDate = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    with mlflow.start_run(run_name=f"llfs_train_{sDate}"):
        # Log training parameters.
        mlflow.log_params(config)

        directory_path = 'data'

        data_loader = create_dataloader(directory_path, tokenizer, batch_size=config['batch_size'],
                                        max_length=config['max_seq_length'], stride=config['max_seq_length'],
                                        num_workers=10, world_size=1, rank=0)

        model = LlamaModel2(vocab_size, embed_dim=config['embed_dim'], 
                            hidden_dim=config['hidden_dim'], num_layers=config['num_layers'], 
                            num_heads=config['num_heads'], dropout=config['dropout'])


  
       # If there are multiple GPUs, wrap the model with nn.DataParallel
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # model = torch.compile(model)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])

    model.train()  # Set model to training mode
    batch_idx = 0
    # Train the model
    for epoch in range(epochs):
        # total_batches = data_loader.dataset.size()
        with tqdm(data_loader, unit="batch") as tepoch:
            for batch, (data, target) in enumerate(tepoch):                
                tepoch.set_description(f"Epoch {epoch}")
                
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)

                output = output.view(-1, vocab_size)
                target = target.view(-1)
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()                
                accuracy = compute_accuracy(output, target)

                if float(loss.item()) < 0.06:
                    # early stop
                    break

                # if batch % 100 == 0:
                loss, current = loss.item(), batch
                mlflow.log_metric("loss", f"{loss:3f}", step=(batch // 100))
                mlflow.log_metric("accuracy", f"{accuracy:3f}", step=(batch // 100))
                print(
                    f"Epoch {epoch} loss: {loss:3f} accuracy: {accuracy:3f} [{current} / {len(data_loader)}]"
                )
                batch_idx += 1

                batch_idx += 1
                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)


    torch.save(model.state_dict(), './llmfs_weights.pth')
    return model, tokenizer


if __name__ == "__main__":
    #git clone --depth=1 --branch=main https://github.com/mlschmitt/classic-books-markdown data && rm -rf data/.git
    model, tokenizer = train(1)

    import IPython
    IPython.embed()

    result = generate_text(model, tokenizer, seed_text="in the beginning ", max_length=100)
    print(result)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    import IPython
    IPython.embed()