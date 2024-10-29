#!/usr/bin/env python
# coding: utf-8
import os
import torch
import tiktoken
import warnings
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
import torch.distributed as dist
import torch.multiprocessing as mp
from importlib.metadata import version
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
# our tools
from llfs_data import create_dataloader
from llfs_model import LlamaModel2
from llfs_config import get_config


pkgs = ["matplotlib",
        "torch",
        "tiktoken"
       ]

for p in pkgs:
    print(f"{p} version: {version(p)}")

torch.manual_seed(129)

def cleanup():
    dist.destroy_process_group()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


if torch.cuda.is_available():
    device_cap = torch.cuda.get_device_capability()
    if device_cap not in ((7, 0), (8, 0), (9, 0)):
        warnings.warn(
            "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
            "than expected."
        )


# Function to compute accuracy
def compute_accuracy(preds, labels):
    # Get the index of the max log-probability
    _, predicted = torch.max(preds, dim=-1)
    correct = (predicted == labels).float()
    accuracy = correct.sum() / len(labels)
    return accuracy.item()


def train_ddp(rank=0, world_size=1, epochs=1):
    setup(rank, world_size)
    device: str = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"setting device to {device}\n")
    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab # Use tokenizer's vocab size
    config = get_config()

    batch_size = config['batch_size']
    embed_dim = config['embed_dim']
    num_heads = config['num_heads']
    hidden_dim =config['hidden_dim']
    num_layers = config['num_layers']
    max_seq_length = config['max_seq_length']
    dropout = config['dropout']

    directory_path = 'data'

    data_loader = create_dataloader(directory_path, tokenizer, batch_size=batch_size,
                                    max_length=max_seq_length, stride=max_seq_length,
                                    num_workers=1, world_size=world_size, rank=rank)

    model = LlamaModel2(vocab_size, embed_dim, hidden_dim, num_layers, num_heads, dropout)

    # If there are multiple GPUs, wrap the model with nn.DataParallel

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # local_rank = int(os.environ["LOCAL_RANK"])
    # model.to("cuda")
    model = DDP(model)


    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    model.train()  # Set model to training mode
    batch_idx = 0

    # Train the model
    for epoch in range(epochs):
        # total_batches = data_loader.dataset.size()
        with tqdm(data_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                tepoch.sampler.set_epoch(epoch) 
                print(f"Sending to device {device}") 
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)

                output = output.view(-1, vocab_size)
                target = target.view(-1)
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()
                
                accuracy = compute_accuracy(output, target)

                if float(loss.item()) < 0.06:
                    break

                batch_idx += 1
                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)


    torch.save(model.state_dict(), './llmfs_weights.pth')

    cleanup()

    # return model, tokenizer

@record
def run_ddp(demo_fn, world_size, epochs):
    mp.spawn(demo_fn,
             args=(world_size, epochs),
             nprocs=world_size,
             join=True)

  
if __name__ == "__main__":
    #git clone --depth=1 --branch=main https://github.com/mlschmitt/classic-books-markdown data && rm -rf data/.git
    epochs = 1

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus

    run_ddp(train_ddp, world_size, epochs)

    import IPython
    IPython.embed()
    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab # Use tokenizer's vocab size
 
    # result = generate_text(model, tokenizer, seed_text="in the beginning ", max_length=100)
    # print(result)

    # print(f'The model has {count_parameters(model):,} trainable parameters')

    import IPython
    IPython.embed()
