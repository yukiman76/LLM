import os
LOCAL_DISK_MOUNT = '/mnt/data'

os.environ['HF_HOME'] = f'{LOCAL_DISK_MOUNT}/hf_cache/'
os.environ['HF_DATASETS_CACHE'] = f'{LOCAL_DISK_MOUNT}/datasets/'

from datasets import load_dataset
from datasets import load_dataset_builder
from psutil._common import bytes2human



print(f"using {LOCAL_DISK_MOUNT}")

def print_dataset_size_if_provided(*args, **kwargs):
  dataset_builder = load_dataset_builder(*args, **kwargs)
  if dataset_builder.info.download_size and dataset_builder.info.dataset_size:
    print(f'{args}: download_size={bytes2human(dataset_builder.info.download_size)}, dataset_size={bytes2human(dataset_builder.info.dataset_size)}')
  else:
    print(f'Dataset size for {args[0]} is not provided by uploader')


# togethercomputer/RedPajama-Data-1T is a clean-room, fully open-source implementation of the LLaMa dataset.
dsn = "tiiuae/falcon-refinedweb"
print_dataset_size_if_provided(dsn)
ds = load_dataset(dsn)
ds.save_to_disk(f'{LOCAL_DISK_MOUNT}/datasets/{dsn.replace("/","_")}')


# allenai/c4  is the processed version of Google's C4 dataset, 
dsn = "allenai/c4"
print_dataset_size_if_provided(dsn)
en = load_dataset("allenai/c4", "en")
en.save_to_disk(f'{LOCAL_DISK_MOUNT}/datasets/{dsn.replace("/","_")}')

dsn = "togethercomputer/RedPajama-Data-1T"
print_dataset_size_if_provided(dsn)
ds = load_dataset(dsn)
ds.save_to_disk(f'{LOCAL_DISK_MOUNT}/datasets/{dsn.replace("/","_")}')


