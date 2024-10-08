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
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"setting device to {device}")


class LlamaModel2(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout,
        )
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output = self.transformer(embedded, embedded)
        output = self.fc(output)
        return output


class LlamaModel_simple(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_heads, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.fc(output)
        return output


def train():
    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab # Use tokenizer's vocab size
    embed_dim = 256  # Size of token embeddings
    num_heads = 8  # Number of attention heads in transformer
    hidden_dim = 2048  # Size of feedforward layer
    num_layers = 12  # Number of transformer layers
    max_seq_length = 512  # Maximum sequence length (context_length)
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
    for epoch in range(8):
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
    return model


def generate_text(model, seed_text, num_tokens):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track the gradients
        tokens = [vocab[token] for token in tokenizer(seed_text)]
        tokens = torch.tensor(tokens).unsqueeze(0).to(device)
        for _ in range(num_tokens):
            output = model(tokens)
            probabilities = nn.functional.softmax(output[0, -1], dim=0)
            next_token = torch.multinomial(probabilities, 1).item()
            tokens = torch.cat([tokens, torch.tensor([[next_token]]).to(device)], dim=1)
        generated_text = ' '.join(vocab.get_itos()[token] for token in tokens[0].cpu().numpy())
        return generated_text


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    #git clone --depth=1 --branch=main https://github.com/mlschmitt/classic-books-markdown data && rm -rf data/.git
    model = train()

    result = generate_text(model, human_input="Generative AI is ", num_tokens=100)
    print(result)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    import IPython
    IPython.embed()
