import glob
import torch
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader, IterableDataset




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

    def __len__(self):
        return  self.i_len

def create_dataloader(sdir, tokenizer, batch_size=4, max_length=256,stride=128, 
                         shuffle=False, drop_last=True, num_workers=0, world_size=2,  
                         rank=1):

    # Create dataset
    dataset = GPTDatasetV2(sdir, tokenizer, max_length, stride)
    sampler = None
    
    # if torch.cuda.device_count() > 1:
    #     sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, 
    #                                  drop_last=False)
       
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=False,
                            drop_last=drop_last, num_workers=num_workers, sampler=sampler)


    return dataloader
