#!/usr/bin/env python
# coding: utf-8
# in order to be able to run this, you must have at least four GPU's with a combined the RAM of 128 gigs or more
# and you must have installed tensorflow-gpu 1.4.0 and keras 2.0.6
# following the readme to install dependencys 
# if you have issues with vRAM adjust batch size and number of epochs accordingly

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
from torch.optim import Adam
from datetime import datetime
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
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12356'
    
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


# if torch.cuda.is_available():
#     device_cap = torch.cuda.get_device_capability()
#     if device_cap not in ((7, 0), (8, 0), (9, 0)):
#         warnings.warn(
#             "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
#             "than expected."
#         )


# Function to compute accuracy
def compute_accuracy(preds, labels):
    # Get the index of the max log-probability
    _, predicted = torch.max(preds, dim=-1)
    correct = (predicted == labels).float()
    accuracy = correct.sum() / len(labels)
    return accuracy.item()


def train_ddp(rank=0, world_size=1, epochs=1):
    setup(rank, world_size)
    print(f"Running DDP with model parallel example on rank {rank}.")
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
    sModelPath = f"llfs_train_{sDate}"

    run_object = mlflow.search_runs(filter_string=f"attributes.run_name = '{sModelPath}'")
    if not run_object.empty:
        run_id = run_object["run_id"]
        mlflow_run = mlflow.start_run(run_id=run_id[0])
    else:
        mlflow_run =  mlflow.start_run(run_name=sModelPath)

    with mlflow_run:
        # Log training parameters.
        mlflow.log_params(config)

        directory_path = 'data'

        data_loader = create_dataloader(directory_path, tokenizer, batch_size=config['batch_size'],
                                        max_length=config['max_seq_length'], stride=config['max_seq_length'],
                                        num_workers=10, world_size=world_size, rank=rank)

        model = LlamaModel2(vocab_size, embed_dim=config['embed_dim'], 
                            hidden_dim=config['hidden_dim'], num_layers=config['num_layers'], 
                            num_heads=config['num_heads'], dropout=config['dropout'])


        model.to(rank)
        model = DDP(model, device_ids=[rank])

        # CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
        # if rank == 0:
        #     # All processes should see same parameters as they all start from same
        #     # random parameters and gradients are synchronized in backward passes.
        #     # Therefore, saving it in one process is sufficient.
        #     torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)


        optimizer = Adam(model.parameters(), lr=config["learning_rate"])

        model.train()  # Set model to training mode
        batch_idx = 0

        # Train the model
        for epoch in range(epochs):
            for batch, (data, target) in enumerate(data_loader):
                data, target = data.to(rank), target.to(rank)
                output = model(data)

                output = output.view(-1, vocab_size)
                target = target.view(-1)
                loss = loss_fn(output, target)

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


        torch.save(model.state_dict(), f"./llmfs_weights_{sDate}.pth")
        # torch.save(model.state_dict(), './llmfs_weights.pth')
        # Save the trained model to MLflow.
        mlflow.pytorch.log_model(model, "model")

    cleanup()

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
