import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        head_size = config.n_embd // config.n_heads

        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.head_size = head_size

        # non-learnable data (do not receive gradients or participate in optimization)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) # (B, T, head_size)
        k = self.key(x) # (B, T, head_size)
        wei = (q @ k.transpose(-2, -1)) * self.head_size ** -0.5 # (B, T, T)

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # T can be smaller than block_size!!!!
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out
    


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config.n_embd, config.dropout)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.block_size = config.block_size

        # emb table for token in vocab
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)

        # emb table for each position
        self.pos_embedding_table = nn.Embedding(config.block_size, config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_blocks)])
        self.ln_f = nn.LayerNorm(config.n_embd) # final layer norm

    def forward(self, idx, targets = None):
        B, T = idx.shape

        # channels = n_embd = 32 (not vocab_size!) so we do not receive logits directly anymore
        token_emb = self.token_embedding_table(idx) # (B, T, C), where B-batch, T-time, C-channel
        pos_emb = self.pos_embedding_table(torch.arange(T, device = self.device)) # (T, C)
        x = token_emb + pos_emb # x holds not just token identities but their positions as well
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, V), where V-vocab_size

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
            idx_cut = idx[:, -self.block_size:] # take only last block_size tokens
            logits, _ = self.forward(idx_cut)
            logits = logits[:, -1, :] # take only last token to predict the next one
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples= 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx