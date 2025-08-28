import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 4 # batches of blocks to process in parallel
block_size = 8 # size of blocks to train on
max_iters = 5000
eval_interval = 500
eval_iters = 50
n_embd = 32
head_size = 16
learning_rate = 1e-3

with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

print(f"dataset length:")
print(f"{len(text)} chars")
print(f"{len(text.split(' '))} words")

chars = sorted(list(set(text)))
vocab_size = len(chars)

# Tokenizer
stoi = {char:i for i,char in enumerate(chars)}
itos = {i:char for char,i in stoi.items()}
encode = lambda s: [stoi[c] for c in s] # string -> list[int]
decode = lambda l: ''.join(itos[i] for i in l) # list[int] -> string

data = torch.tensor(encode(text), dtype=torch.long)

# Train and Validation Data Sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

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

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'validation']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHead(4, n_embd // 4)
        self.lang_model_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, index, targets=None):
        B, T = index.shape
        
        # gets embedding from token index
        token_embd = self.token_embedding_table(index) # (B, T, n_embd)
        position_embd = self.position_embedding_table(torch.arange(T)) # (T, C)
        x = token_embd + position_embd
        
        # self attention
        x = self.sa_heads(x)
        
        # returns (batch, time {which char of the batch}, channel {vocab_size})
        logits = self.lang_model_head(x)
        
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
            index_crop = index[:, -block_size:]
            logits, loss = self(index_crop)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # attention scores
        # note scaled attention to lower variance and suit softmax
        weights = q @ k.transpose(-2, -1) * (C**-0.5) # (B, T, head_size) @ (B, head_size, T) ==> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        
        v = self.value(x)
        out = weights @ v
        return out

class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
        
model = BigramLanguageModel()

x, y = get_batch('train')
logits, loss = model(x, y)

index = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(index, max_new_tokens=100)[0].tolist()))


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

batch_size = 32

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {step}: Train Loss {losses['train']:.4f}, Validation Loss {losses['validation']:4f}")
    
    # get sample of data
    x, y = get_batch('train')
    
    # evaluate loss and backprop
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(decode(model.generate(index, max_new_tokens=100)[0].tolist()))
