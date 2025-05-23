#!/usr/bin/env python
# coding: utf-8
from torch import nn


# TODO: add layer normalization before and after the attention layers.
# TODO: add positional encoding to give the model a sense of token positions
# TODO: add residual connections arround the transformers blocks to make the gradient
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
