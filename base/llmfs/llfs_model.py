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

class LlamaModel2(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.Transformer(
            batch_first=True, # remove warning
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
