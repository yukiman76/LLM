import time
import math
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import tiktoken
import lightning as L

BATCH_SIZE = 16
CONTEXT_LENGTH = 512

train_batch_size = 16  # training batch size
eval_batch_size = 8  # evaluation batch size
# context_length = 512  # number of tokens processed in a single batch
train_split = 0.7  # percentage of data to use from total data for training


# used to define size of embeddings
d_model = 768
n_heads = 4  # number of self-attention heads. should be divisible with d_model
n_layers = 8  # number of gpt blocks/layers


# training setup
epochs = 2000
eval_steps = 500  # perform evaluation in every n steps
lr = 1e-3


class GPTDataSet(Dataset):
    def __init__(self, tokens, batch_size, context_length) -> None:
        self.tokens = tokens
        self.batch_size = batch_size
        self.context_length = context_length
        self.current_position = 0

    def __len__(self):
        return len(self.tokens) - self.batch_size

    def __getitem__(self, idx):
        b, c = self.batch_size, idx

        start_pos = self.current_position
        end_pos = self.current_position + b * c + 1

        # if the batch exceeds total length, get the data till last token
        # and take remaining from starting token to avoid always excluding some data
        add_data = (
            -1
        )  # n, if length exceeds and we need `n` additional tokens from start
        if end_pos > len(self.tokens):
            add_data = end_pos - len(self.tokens)
            end_pos = len(self.tokens)

        d = self.tokens[start_pos:end_pos]
        if add_data != -1:
            d = torch.cat([d, self.tokens[:add_data]])

        x = (d[:-1]).view(b, c)  # inputs
        y = (d[1:]).view(b, c)  # targets

        self.current_position += b * c  # set the next position
        if self.current_position > len(self.tokens) - 1:
            self.current_position = 0
        print(x, y)
        return x, y


class GPT2DataModule(L.LightningDataModule):
    def __init__(self, data_file: str):
        super().__init__()
        self.data_file = data_file
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.tokenizer.n_vocab

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        text = open(self.data_file, "r").read()  # load all the data as simple string
        # convert our text data into tokenized tensor
        self.data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            # split data into trian and eval
            n_data = len(self.data)
            self.gpt2_train = GPTDataSet(self.data[: int(n_data * train_split)],
                                       batch_size=BATCH_SIZE,
                                       context_length=CONTEXT_LENGTH)


            self.gpt2_val = GPTDataSet(self.data[int(n_data * train_split) :],
                                       batch_size=BATCH_SIZE,
                                       context_length=CONTEXT_LENGTH)


        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.gpt2_test = GPTDataSet(self.data,
                                      batch_size=BATCH_SIZE,
                                      context_length=CONTEXT_LENGTH)

    def train_dataloader(self):
        return DataLoader(self.gpt2_train, batch_size=BATCH_SIZE, num_workers=31)

    def val_dataloader(self):
        return DataLoader(self.gpt2_val, batch_size=BATCH_SIZE, num_workers=31)

    def test_dataloader(self):
        return DataLoader(self.gpt2_test, batch_size=BATCH_SIZE, num_workers=31)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert n_heads * self.head_dim == d_model

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, inputs: torch.Tensor):
        B, seq_length, d_model = inputs.shape

        # Project the input embeddings into Q, K, and V
        Q = (
            self.query(inputs)
            .view(B, seq_length, self.n_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        K = (
            self.key(inputs)
            .view(B, seq_length, self.n_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        V = (
            self.value(inputs)
            .view(B, seq_length, self.n_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )

        # Apply mask to prevent attention to future tokens
        mask = (
            torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
            .bool()
        )
        attention_scores = attention_scores.masked_fill(mask, float("-inf"))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        # Compute the weighted sum of the values
        attention_output = torch.matmul(self.dropout(attention_weights), V)

        # Concatenate heads and put them back to the original shape
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(B, seq_length, d_model)

        # Apply the final linear transformation
        out = self.fc_out(attention_output)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, context_length, d_model) -> None:
        super().__init__()
        # Create a matrix of shape (context_length, d_model) to store the positional encodings
        pe = torch.zeros(context_length, d_model)

        # Create a vector with positions [0, 1, 2, ..., context_length-1] of shape (context_length, 1)
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)

        # Create a vector with the divisor terms based on the dimension
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Compute the positional encodings using sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, context_length, d_model)

        # Register pe as a buffer, so it is not considered a parameter but is part of the module's state
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add the positional encodings to the input embeddings
        return x + self.pe[:, : x.size(1), :]


class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.att = MultiHeadAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
        self.fcn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, logits):
        att_logits = self.att(logits)
        adn_logits = self.ln1(logits + att_logits)
        logits = self.dropout(adn_logits)
        logits = self.fcn(logits)
        logits = self.ln2(logits + adn_logits)
        return logits


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)  # word token embeddings
        self.wpe = PositionalEncoding(
            CONTEXT_LENGTH, d_model
        )  # word position encodings
        self.blocks = nn.ModuleList(
            [GPTBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.linear1 = nn.Linear(d_model, vocab_size)

        #parameter sharing
        self.wte.weight = self.linear1.weight

    def forward(self, inputs, targets=None):
        logits = self.wte(inputs)  # dim -> batch_size, sequence_length, d_model
        logits = self.wpe(logits)
        for block in self.blocks:
            logits = block(logits)
        logits = self.linear1(logits)
        loss = None
        if targets != None:
            batch_size, sequence_length, d_model = logits.shape
            # to calculate loss for all token embeddings in a batch
            # kind of a requirement for cross_entropy
            logits = logits.view(batch_size * sequence_length, d_model)
            targets = targets.view(batch_size * sequence_length)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, inputs, max_new_tokens):
        # this will store the model outputs along with the initial input sequence
        # make a copy so that it doesn't interfare with model
        output = inputs.clone()
        for _ in range(max_new_tokens):
            current_seq_length = inputs.size(1)
            # Truncate inputs if it exceeds context_length
            if current_seq_length > CONTEXT_LENGTH:
                inputs = inputs[:, -CONTEXT_LENGTH:]
            # we only pass targets on training to calculate loss
            logits, _ = self(inputs)
            # for all the batches, get the embeds for last predicted sequence
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            # get the probable token based on the input probs
            idx_next = torch.multinomial(probs, num_samples=1)

            inputs = torch.cat([inputs, idx_next], dim=1)
            output = torch.cat([output, idx_next], dim=1)
        return [tokenizer.decode(out.tolist()) for out in output]


class LightningTransformer(L.LightningModule):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super().__init__()
        self.model = GPT(vocab_size=vocab_size,
                         d_model=d_model,
                         n_heads=n_heads,
                         n_layers=n_layers)

    def forward(self, inputs, target):
        return self.model(inputs, target)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.1)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000, eta_min=lr*0.1)
        return [optimizer], [lr_scheduler]


if __name__ == "__main__":
    sfile = 'data.txt'
    # dataloader = DataLoader(train_data, train_batch_size, context_length)
    dataloader = GPT2DataModule(sfile)
    # eval_loader = DataLoader(eval_data, eval_batch_size, context_length)
    # m = torch.compile(m)
    model = LightningTransformer(dataloader.vocab_size, d_model, n_heads, n_layers)
    trainer = L.Trainer(accelerator="auto", devices="auto", strategy="auto")
    trainer.fit(model=model, train_dataloaders=dataloader)

    torch.save(model, 'GPT_light.pt')

    # saying to torch that do not store gradients for whatever we do below
    with torch.no_grad():
        input = torch.tensor(
            tokenizer.encode("why is the sky blue "), dtype=torch.long,
        ).unsqueeze(0)
        print(model.generate(input, max_new_tokens=500)[0])
