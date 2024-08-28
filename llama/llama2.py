import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.optim import Adam
from data import load_abstracts, load_local_data # we get test data from here


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the dataset
class LlamaDataset(Dataset):
    def __init__(self, text, sequence_length):
        self.text = text
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.text) - self.sequence_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.text[idx:idx+self.sequence_length]),
            torch.tensor(self.text[idx+1:idx+self.sequence_length+1]),
        )

class LlamaModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_heads, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size,
            dropout=dropout,
        )
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output = self.transformer(embedded, embedded)
        output = self.fc(output)
        return output


# Use the trained model to generate new text
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


if __name__ == "__main__":
    big_text = load_abstracts("LLMs Generative AI", number_paper=400)
    # Lowercase the text
    text = big_text.lower()
    # Define the tokenizer
    tokenizer = get_tokenizer('basic_english')
    # Tokenize the text
    tokenized_text = [list(tokenizer(text))]
    # Build the vocabulary from the tokenized text
    vocab = build_vocab_from_iterator(tokenized_text)

    # Numericalize the text
    numericalized_text = [vocab[token] for token in tokenized_text[0]]
    # Create the dataset and dataloader
    sequence_length = 30
    dataset = LlamaDataset(numericalized_text, sequence_length)
    dataloader = DataLoader(dataset, batch_size=128)

    # Initialize the model and the optimizer
    model = LlamaModel(len(vocab), embed_size=128, hidden_size=256,
                       num_layers=2, num_heads=8, dropout=0.1)

    # If there are multiple GPUs, wrap the model with nn.DataParallel
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(80):
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = nn.functional.cross_entropy(y_pred.view(-1, len(vocab)), y.view(-1))
            # grad can be implicitly created only for scalar outputs
            loss.sum().backward()
            optimizer.step()
        print(f'Epoch {epoch}, Loss {loss.item()}')
        if float(loss.item()) < 0.06:
            break

    result = generate_text(model, human_input="Generative AI is ", num_tokens=100)
    print(result)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    print(f'The model has {len(vocab)} tokens')
