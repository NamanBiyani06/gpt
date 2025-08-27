# %%
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

print(text[:100])

# %%
print(f"dataset length:")
print(f"{len(text)} chars")
print(f"{len(text.split(' '))} words")

# %%
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(chars)
print(vocab_size)

# %% [markdown]
# Tokenizer

# %%
stoi = {char:i for i,char in enumerate(chars)}
itos = {i:char for char,i in stoi.items()}
encode = lambda s: [stoi[c] for c in s] # string -> list[int]
decode = lambda l: ''.join(itos[i] for i in l) # list[int] -> string

print(decode(encode("Naman")))

# %%
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])

# %% [markdown]
# Train and Validation Data Sets

# %%
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# %%
block_size = 8
print(train_data[:block_size + 1])
print(decode(train_data[:block_size + 1].tolist()))

# %%
x = train_data[:block_size]
y = train_data[1:block_size + 1]
print(x)
print(y)

for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    print(f"context: {context} --> target: {target}")

# %%
batch_size = 4 # batches of blocks to process in parallel
block_size = 8 # size of blocks to train on

def get_batch(split):
    match split:
        case 'train':
            data = train_data
        case 'validation':
            data = val_data
    
    indices = torch.randint(0, len(data) - block_size, size=(batch_size, ))
    
    x = torch.stack([data[i:i + block_size] for i in indices])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in indices])
    
    return x, y

x, y = get_batch('train')
print("inputs:")
print(x)
print("targets:")
print(y)
    

# %%
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, index, targets=None):
        # returns (batch, time {which char of the batch}, channel {vocab_size})
        logits = self.token_embedding_table(index)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(index)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

model = BigramLanguageModel(vocab_size)
logits, loss = model(x, y)
print(logits.shape)
print(loss)

index = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(index, max_new_tokens=100)[0].tolist()))


# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# %%
batch_size = 32

for steps in range(10000):
    x, y = get_batch('train')
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
print(loss.item())

# %%
print(decode(model.generate(index, max_new_tokens=100)[0].tolist()))


