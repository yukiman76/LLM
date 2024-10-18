#!/usr/bin/env python
# coding: utf-8
import os
import glob
import torch
import tiktoken
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from importlib.metadata import version
from torch.utils.data import Dataset, DataLoader, IterableDataset
# our tools
from llfs_infrence import generate_text, count_parameters
from llfs_model import LlamaModel_simple, LlamaModel2

pkgs = ["matplotlib",
        "torch",
        "tiktoken"
       ]
for p in pkgs:
    print(f"{p} version: {version(p)}")

torch.manual_seed(129)


class GPTDatasetV2(IterableDataset):
    def __init__(self, directory, tokenizer, max_length, stride):
        self.directory = directory
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.i_len = len(glob.glob(f"{self.directory}/**/*.md", recursive=True))

    def __iter__(self):
        docs = glob.glob(f"{self.directory}/**/*.md", recursive=True)
        with tqdm(docs, unit="document") as tqdocs:
            for filename in tqdocs:
                with open(filename, 'r') as f:
                    txt = f.read()
                    token_ids = self.tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
                    for i in range(0, len(token_ids) - self.max_length, self.stride):
                        input_chunk = token_ids[i:i + self.max_length]
                        target_chunk = token_ids[i + 1: i + self.max_length + 1]
                        yield torch.tensor(input_chunk), torch.tensor(target_chunk)

    def size(self):
        return self.i_len


def create_dataloader_v2(sdir, tokenizer, batch_size=4, max_length=256,
                         stride=128, shuffle=False, drop_last=True, num_workers=0):

    # Create dataset
    dataset = GPTDatasetV2(sdir, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, num_workers=num_workers)

    return dataloader


    
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
    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab # Use tokenizer's vocab size
    batch_size = 8
    # TODO: refactor to a prams class 
    if device == 'cuda':
        # -- ver 2
        embed_dim = 256  # Size of token embeddings
        num_heads = 8  # Number of attention heads in transformer
        hidden_dim = 2048  # Size of feedforward layer
        num_layers = 12  # Number of transformer layers
        max_seq_length = 512  # Maximum sequence length (context_length)
        dropout=0.1

        # -- ver 3, LLama3 Nvidia 2x V100  32G
        if "V100" in torch.cuda.get_device_name(0):
            print("Using Nvidia 2x V100 GPU config")
            embed_dim = 256 # Size of token embeddings
            num_heads = 32 # Number of attention heads in transformer
            hidden_dim = 2048  # Size of feedforward layer
            num_layers = 32 # Number of transformer layers
            max_seq_length = 1024 # vMEM poor -  2048  # Maximum sequence length (context_length)
            dropout=0.1

        if "4090" in torch.cuda.get_device_name(0):
            print("Using Nvidia 4x 4090 GPU config")
            batch_size = 10
            # -- ver 3, LLama3 nvidia 4 x 4090 24G
            embed_dim = 256  # Size of token embeddings
            num_heads = 32  # Number of attention heads in transformer
            hidden_dim = 4096  # Size of feedforward layer
            num_layers = 32  # Number of transformer layers
            max_seq_length = 512  # Maximum sequence length (context_length)
            dropout=0.1

    else:
        print("Using default")
        # -- ver 3, LLama3 nvidia 4 x 4090 24G
        embed_dim = 256  # Size of token embeddings
        num_heads = 32  # Number of attention heads in transformer
        hidden_dim = 4096  # Size of feedforward layer
        num_layers = 32  # Number of transformer layers
        max_seq_length = 512  # Maximum sequence length (context_length)
        dropout=0.1 

    directory_path = 'data'

    data_loader = create_dataloader_v2(directory_path, tokenizer, batch_size=batch_size,
                                       max_length=max_seq_length, stride=max_seq_length)

    model = LlamaModel2(vocab_size, embed_dim, hidden_dim, num_layers, num_heads, dropout)


    # If there are multiple GPUs, wrap the model with nn.DataParallel
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # model = torch.compile(model)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    model.train()  # Set model to training mode
    batch_idx = 0
    # Train the model
    for epoch in range(epochs):
        total_batches = data_loader.dataset.size()
        with tqdm(data_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)

                output = output.view(-1, vocab_size)
                target = target.view(-1)
                # loss = nn.functional.cross_entropy(output.view(-1, vocab_size), target.view(-1))
                loss = criterion(output, target)

                # import IPython
                # IPython.embed()

                loss.backward()
                optimizer.step()
                
                accuracy = compute_accuracy(output, target)

                # print(f'Epoch {epoch}, Loss {loss.item()}')
                if float(loss.item()) < 0.06:
                    break

                # if batch_idx % 100 == 99:  # Print every 100 batches
                #     print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{total_batches}], Loss: {running_loss/100:.4f}, Accuracy: {running_accuracy/100:.4f}')
                #     running_loss = 0.0
                #     running_accuracy = 0.0

                batch_idx += 1
                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)


    torch.save(model.state_dict(), './llmfs_weights.pth')
    return model, tokenizer


if __name__ == "__main__":
    #git clone --depth=1 --branch=main https://github.com/mlschmitt/classic-books-markdown data && rm -rf data/.git
    model, tokenizer = train(8)

    import IPython
    IPython.embed()

    result = generate_text(model, tokenizer, seed_text="in the beginning ", max_length=100)
    print(result)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    import IPython
    IPython.embed()