import os
LOCAL_DISK_MOUNT = '/mnt/data'
if os.path.exists(LOCAL_DISK_MOUNT):
    os.environ['HF_HOME'] = f'{LOCAL_DISK_MOUNT}/hf_cache/'
    os.environ['HF_DATASETS_CACHE'] = f'{LOCAL_DISK_MOUNT}/datasets/'

import glob
import torch
import tiktoken
from tqdm import tqdm
# from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, IterableDataset
# from torch.utils.data.distributed import DistributedSample
from datasets import concatenate_datasets, load_dataset

# distributedSampler

class GPTDatasetV2(IterableDataset):
    def __init__(self, directory, tokenizer, max_length, stride):
        self.directory = directory
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.docs = glob.glob(f"{self.directory}/**/*.md", recursive=True)
        self.i_len = len(self.docs)

    def __iter__(self):
        with tqdm(self.docs, unit="document") as tqdocs:
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

    def __len__(self):
        return  self.i_len

def create_dataloader(sdir, tokenizer, batch_size=4, max_length=256, stride=128, 
                         shuffle=False, drop_last=True, num_workers=0, world_size=2,  
                         rank=1):

    # Create dataset
    books_ds = GPTDatasetV2(sdir, tokenizer, max_length, stride)
    sampler = None
    
    # if torch.cuda.device_count() > 1:
    #     sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, 
    #                                  drop_last=False)

    # owt =  load_dataset("openwebtext") 
    # en = load_dataset("allenai/c4", "en", split="train", streaming=True)
    # en = en.take(1_000)
    # Concatenate both datasets
    # concatenated = concatenate_datasets([books_ds, en]) 
    
    # Create dataloader
    dataloader = DataLoader(books_ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
                            drop_last=drop_last, num_workers=num_workers, sampler=sampler)


    return dataloader


if __name__ == "__main__":
    directory_path = 'data'
    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab # Use tokenizer's vocab size
    batch_size = 8
    embed_dim = 256  # Size of token embeddings
    num_heads = 8  # Number of attention heads in transformer
    hidden_dim = 2048  # Size of feedforward layer
    num_layers = 12  # Number of transformer layers
    max_seq_length = 512  # Maximum sequence length (context_length)
    dropout=0.1

    data_loader = create_dataloader(directory_path, tokenizer, batch_size=batch_size,
                                       max_length=max_seq_length, stride=max_seq_length) 