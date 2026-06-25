import torch
import torch.nn as nn
from torch.nn import functional as F
import utils
import numpy as np


torch.manual_seed(1337)


class LayerNorm1d:
    def __init__(self, dim, eps=1e-5):
        self.eps=eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)


    def __call__(self, x):
        x_mean = x.mean(1,keepdim=True)
        x_var = x.var(1, keepdim=True)
        x_hat = (x - x_mean) / torch.sqrt(x_var + self.eps)

        self.out = self.gamma + x_hat + self.beta
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size,context_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2,-1) / np.sqrt(C)
        wei = wei.masked_fill(self.tril[n_head:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        wei = self.dropout(wei)

        out = wei @ v
        return out
    

class MultiHeadAttantion(nn.Module):
    def __init__(self, num_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        return out
    

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed, 4 * n_embed), nn.ReLU(),nn.Linear(4 * n_embed, n_embed), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttantion(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class TransformerLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(context_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-context_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


if __name__ == '__main__':
    text = utils.load_text("/home/fedor/scicodes/ml/nano-gpt-tutorial/data/Pushkin.txt")
    itos, _, encode, decode = utils.get_vocabulary(text)
    train_data, val_data = utils.get_train_val_split(0.9, text)

    vocab_size = len(itos)
    context_size = 124
    batch_size = 32
    n_embed = 64
    n_head = 6
    n_layer = 6

    dropout = 0.2

    max_iters = 10000
    eval_interval = 500
    learning_rate = 3e-4
    device = 'cuda'

    eval_iters = 200

    model = TransformerLanguageModel()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        if iter % eval_interval == 0:
            losses = utils.estimate_loss(model, train_data, val_data, eval_iters, batch_size, context_size, device)
            print(f"step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

        xb, yb = utils.get_batch(data=train_data, context_size=context_size, batch_size=batch_size, device=device)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.ones((1,1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))