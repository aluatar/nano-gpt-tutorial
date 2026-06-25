import torch
import torch.nn as nn
from torch.nn import functional as F
import utils

torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.tokent_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        logits = self.tokent_embedding_table(idx)

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
            logits, _ = self(idx)
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
    context_size = 10
    batch_size = 32

    max_iters = 90000
    eval_interval = 300
    learning_rate = 1e-4
    device = 'cuda'

    eval_iters = 200

    model = BigramLanguageModel(vocab_size)
    m = model.to(device)

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
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))