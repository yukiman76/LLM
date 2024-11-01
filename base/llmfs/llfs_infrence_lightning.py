import os
import torch
import tiktoken
import lightning as L
from llfs_model import LlamaModel2
from llfs_config import get_config

# Set up local disk cache if necessary
LOCAL_DISK_MOUNT = '/mnt/data'
if os.path.exists(LOCAL_DISK_MOUNT):
    os.environ['HF_HOME'] = f'{LOCAL_DISK_MOUNT}/hf_cache/'
    os.environ['HF_DATASETS_CACHE'] = f'{LOCAL_DISK_MOUNT}/datasets/'

class LlamaLightningModule(L.LightningModule):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.model = LlamaModel2(
            vocab_size,
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        )
        self.config = config

    def forward(self, x):
        return self.model(x)

def load_model(checkpoint_path, config, vocab_size):
    """ Load the trained model from checkpoint """
    model = LlamaLightningModule(config=config, vocab_size=vocab_size)
    model.load_state_dict(torch.load(checkpoint_path, 
                                     map_location='cuda' if torch.cuda.is_available() else 'cpu',
                                     weights_only=True))
    model.eval()
    return model

def generate_text(model, tokenizer, seed_text, max_length=100):
    """ Generate text given a seed text and model """
    tokens = tokenizer.encode(seed_text)
    input_ids = torch.tensor(tokens).unsqueeze(0)
    input_ids = input_ids.to('cuda' if torch.cuda.is_available() else 'cpu')

    output_tokens = input_ids.clone()

    # Generate tokens one by one
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(output_tokens)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            output_tokens = torch.cat([output_tokens, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == tokenizer.eot_token:
                print("eot_token reached")
                break

    generated_text = tokenizer.decode(output_tokens.squeeze().tolist())
    return generated_text

def main(checkpoint_path="llmfs_weights.pth", seed_text="Once upon a time"):
    # Load the configuration and tokenizer
    config = get_config()
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokenizer.pad_token = tokenizer.eot_token
    tokenizer.padding_side = "right"  
    tokenizer.eos_token_id = tokenizer.eot_token 

    vocab_size = tokenizer.n_vocab

    # Load the trained model
    model = load_model(checkpoint_path, config, vocab_size)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate text
    generated_text = generate_text(model, tokenizer, seed_text=seed_text, max_length=100)
    print(f"Generated Text:\n{generated_text}")

if __name__ == "__main__":
    m_output_dir = input("What is the model file : ")
    if m_output_dir == "":
        m_output_dir  = "llmfs_weights_11-01-2024-07-47-47.pth" 
    m_output_dir = m_output_dir.lstrip().rstrip()

    main(checkpoint_path=m_output_dir)

    import IPython
    IPython.embed()