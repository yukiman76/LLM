import torch

def get_config():
    config = {
        'batch_size': 8,
        'embed_dim': 256,  # Size of token embeddings
        'num_heads': 8,  # Number of attention heads in transformer
        'hidden_dim': 2048,  # Size of feedforward layer
        'num_layers': 12,  # Number of transformer layers
        'max_seq_length': 512,  # Maximum sequence length (context_length)
        'dropout': 0.1,
    }
    sdevice_name = torch.cuda.get_device_name(0) 
        # -- ver 3, LLama3 Nvidia 2x V100  32G
    if "V100" in sdevice_name:
        print("Using Nvidia V100 GPU config")
        config['num_heads'] = 32 # Number of attention heads in transformer
        config['hidden_dim'] = 4096  # Size of feedforward layer
        config['num_layers'] = 32 # Number of transformer layers
        config['max_seq_length'] = 512 # Maximum sequence length (context_length)

    if "4090" in sdevice_name:
        print("Using Nvidia 4090 GPU config")
        config['batch_size'] = 10
        # -- ver 3, LLama3 nvidia 4 x 4090 24G
        config['num_heads'] = 32  # Number of attention heads in transformer
        config['hidden_dim'] = 4096  # Size of feedforward layer
        config['num_layers'] = 32  # Number of transformer layers

    if "H100" in torch.cuda.get_device_name(0):
        print("Using Nvidia H100 GPU config")
        config['batch_size'] = 8
        config['embed_dim'] = 256  # Size of token embeddings
        config['num_heads'] = 32  # Number of attention heads in transformer
        config['hidden_dim'] = 4096  # Size of feedforward layer
        config['num_layers'] = 32  # Number of transformer layers
        config['max_seq_length'] = 2048  # Maximum sequence length (context_length)

    return config
