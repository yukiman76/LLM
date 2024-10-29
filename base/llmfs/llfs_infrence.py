#!/usr/bin/env python
# coding: utf-8
import torch
import tiktoken
from torch import nn
from importlib.metadata import version
# Our tools
from llfs_model import LlamaModel2
from llfs_config import get_config


pkgs = ["torch",
        "tiktoken"
       ]
for p in pkgs:
    print(f"{p} version: {version(p)}")

device: str = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

torch.manual_seed(129)


def generate_text(model, tokenizer, seed_text, max_length=100):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track the gradientsi
        input_ids = tokenizer.encode(seed_text)
        input_ids_tensor = torch.tensor([input_ids]).to(device)
        generated_tokens = input_ids[:]
        for _ in range(max_length):
            output = model(input_ids_tensor)
            probabilities = nn.functional.softmax(output[0, -1], dim=0)
            next_token = torch.multinomial(probabilities, 1).item()
            generated_tokens.append(next_token)
            # if next_token == tokenizer.eos_token_id:
            #     break
            if len(generated_tokens) >= max_length:
                break
            input_ids_tensor = torch.cat([input_ids_tensor, torch.tensor([[next_token]]).to(device)], dim=1)

        return tokenizer.decode(generated_tokens)


def count_parameters(model):
    i = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'The model has {i:,} trainable parameters')
    return i


if __name__ == "__main__":
    device: str = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"setting device to {device}\n\n")

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
    model = LlamaModel2(vocab_size, embed_dim, hidden_dim, num_layers, num_heads, dropout)
    model.load_state_dict(torch.load('llmfs_weights.pth', weights_only=True))
    model = model.to(device)

    print(f'The model has {count_parameters(model):,} trainable parameters\n')

    result = generate_text(model, tokenizer, seed_text="in the begining ", max_length=100)
    print(result)
    print("\n\n")
     
    import IPython
    IPython.embed()
