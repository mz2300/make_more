import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

block_size = 8
batch_size = 32
steps = 100
eval_interval = 1_000
num_eval_batches = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 5e-3

print(f'Using {device} device')

# open the file and read the content
with open('data/tiny_shakespeare.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()


# create vocab 
unique_symbols = sorted(list(set(text)))
vocab_size = len(unique_symbols)


# create a dictionary to map symbols to indices and vice versa
stoi = {s: i for i, s in enumerate(unique_symbols)}
itos = {i: s for s, i in stoi.items()}

encode = lambda s: [stoi[ch] for ch in s]  # symbols to tokens
decode = lambda l: ''.join([itos[i] for i in l])  # tokens to symbols


# convert the text to tensor
data = torch.tensor(encode(text), dtype = torch.long)


# train test split
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x_batch = torch.stack([data[i:i+block_size] for i in ix])
    y_batch = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    return x_batch, y_batch


class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        # idx in our case is [batch, block_szie]
        # so logits is gonna be [batch, block_szie, vector_dim], where vector_dim  = vocab_size
        logits = self.token_embedding_table(idx) # (B, T, C), where B-batch, T-time, C-channel

        if targets == None:
            loss = None
        else:
            # logits (B, T, C)
            # targets (B, T)
            # but F.cross_entropy expects input of shape (batch, C, <all other dims>) ---> error
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(-1))
            
        return logits, loss


    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(idx)
            logits = logits[:, -1, :] # take only last token to predict the next one
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples= 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx
    

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(num_eval_batches)
        for n_batch in range(num_eval_batches):
            x_batch, y_batch = get_batch(split)
            _, loss = m(x_batch, y_batch)
            losses[n_batch] = loss.item()
        out[split] = losses.mean().item()
    m.train()
    return out


# training loop
m = BigramModel(vocab_size)
m.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr = lr)
for step in range(steps):

    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f'Step {step}, Train loss: {losses["train"]:.3f}, Val loss: {losses["val"]:.3f}') 

    x_batch, y_batch = get_batch('train')
    logits, loss = m(x_batch, y_batch)

    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()


start_token = torch.zeros((1, 1), dtype= torch.int, device = device)
gen_text = m.generate(start_token, 100)
print(decode(gen_text[0].tolist()))