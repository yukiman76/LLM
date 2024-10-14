#!/usr/bin/env python
# coding: utf-8
import os
import glob
import torch
import tiktoken
from torch import nn
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

    def __iter__(self):
        for filename in glob.glob(f"{self.directory}/**/*"):
            print(filename)
            with open(filename, 'r') as f:
                txt = f.read()
                token_ids = self.tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
                for i in range(0, len(token_ids) - self.max_length, self.stride):
                    input_chunk = token_ids[i:i + self.max_length]
                    target_chunk = token_ids[i + 1: i + self.max_length + 1]
                    yield torch.tensor(input_chunk), torch.tensor(target_chunk)

def create_dataloader_v2(sdir, tokenizer, batch_size=4, max_length=256,
                         stride=128, shuffle=False, drop_last=True, num_workers=0):

    # Create dataset
    dataset = GPTDatasetV2(sdir, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, num_workers=num_workers)

    return dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"setting device to {device}")


def train(epochs=1):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab # Use tokenizer's vocab size
    # -- ver 2
    # embed_dim = 256  # Size of token embeddings
    # num_heads = 8  # Number of attention heads in transformer
    # hidden_dim = 2048  # Size of feedforward layer
    # num_layers = 12  # Number of transformer layers
    # max_seq_length = 512  # Maximum sequence length (context_length)
    # dropout=0.1
    # -- ver 3, LLama3
    embed_dim = 256 # Size of token embeddings
    num_heads = 32 # Number of attention heads in transformer
    hidden_dim = 4096  # Size of feedforward layer
    num_layers = 32 # Number of transformer layers
    max_seq_length = 2048  # Maximum sequence length (context_length)
    dropout=0.1


    directory_path = 'data'

    data_loader = create_dataloader_v2(directory_path, tokenizer, batch_size=8,
                                       max_length=max_seq_length, stride=max_seq_length)


    model = LlamaModel2(vocab_size, embed_dim, hidden_dim, num_layers, num_heads, dropout)

    device: str = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')


    # If there are multiple GPUs, wrap the model with nn.DataParallel
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        for batch in data_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = nn.functional.cross_entropy(y_pred.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch}, Loss {loss.item()}')
        if float(loss.item()) < 0.06:
            break

    torch.save(model, './llmfs.pt')
    return model, tokenizer


if __name__ == "__main__":
    #git clone --depth=1 --branch=main https://github.com/mlschmitt/classic-books-markdown data && rm -rf data/.git
    model, tokenizer = train(8)

    import IPython
    IPython.embed()

    result = generate_text(model, tokenizer, seed_text="in the begining ", max_length=100)
    print(result)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    import IPython
    IPython.embed()
