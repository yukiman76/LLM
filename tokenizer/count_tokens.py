#!/usr/bin/env python
# coding: utf-8
import os
import glob
import torch
import tiktoken

def count_tokens(tokenizer, sPath):
    i = 0
    for filename in glob.glob(f"{sPath}/**/*"):
        print(filename)
        with open(filename, 'r') as f:
            token_ids = tokenizer.encode(f.read(),
                                         allowed_special={"<|endoftext|>"})
            i += len(token_ids)

    return i

if __name__ == "__main__":
    #git clone --depth=1 --branch=main https://github.com/mlschmitt/classic-books-markdown data && rm -rf data/.git
    tokenizer = tiktoken.get_encoding("cl100k_base")
    x = count_tokens(tokenizer, './data')
    print(f"{x:,}")
