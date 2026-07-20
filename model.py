"""
remake the Transformer more systematically relying on Karpathy tutorial and github
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int
    context_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 256
    drpout: float = 0.1
    bias: bool = True

class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = 1e-5


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
    

class CausalSelfAttantion(nn.Module):
# It uses flashattention
    def __init__(self, config: ModelConfig):
        super().__init__()
        # check if embedding dimentionality fits the number of the heads
        if config.n_embed % config.n_head != 0:
            raise ValueError(f"In this nanomodel embedding dimentions must be divisible by the number of attantion heads")
        
        # Here we stack Keys, Qeries, and Values projection matricies into one extended matrix to compute kyes, quaries, and values at once with one matrix operation
        # it's for Optimisation 
        self.attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)

        # Linear transfornation of attantion operation output
        self.proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)

        # initialization dropout to attention correction dX' to the residual connection; regularization to minimize overfitting 
        self.resid_dropout = nn.Dropout(config.drpout)

        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.dropout = config.drpout


    def forward(self, X: torch.Tensor) -> torch.Tensor:

        B, T, C = X.size() # batch size, current sequence size (not merely context size!!!), embedding dim

        # calculate query, key, value tensors at once getting  stack of 3 * n_embed - dimentional vectors, 
        # then split in into n_embed-dimentional vector stac -- 3-rank tensor all in all
        q, k, v = self.attn(X).split(self.n_embed, dim=2)

        # reshape key, value, query 3-rank tensors into 4-rank tensor by dividing it into separate heads inputs 
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, n_head, T, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, n_head, T, head_size)

        Y = F.scaled_dot_product_attention(
            query=q, key=k, value=v,
            attn_mask=None, 
            dropout_p=self.dropout if self.training else 0,
            is_causal=True
        )

        Y = Y.transpose(1,2).contiguous().view(B, T, C)

        Y = self.resid_dropout(self.proj(Y))

        return Y
    

class MultuLayerPerceptron(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embed, 4 * config.n_embed, bias=config.bias)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(4 * config.n_embed, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.drpout)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.fc(X)
        X = self.gelu(X)
        X = self.proj(X)
        X = self.dropout(X)

        return X
    

class Block(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embed, config.bias)
        self.attn = CausalSelfAttantion(config)
        self.ln_2 = LayerNorm(config.n_embed, config.bias)
        self.mlp = MultuLayerPerceptron(config)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X + self.attn(self.ln_1(X))
        X = X + self.mlp(self.ln_2(X))

        return X
    

class Transformer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        # checks
        if config.vocab_size is None or config.vocab_size == 0.:
            raise ValueError("Vocabulary should be non-empty")
        if config.context_size is None or config.context_size == 0:
            raise ValueError("Context window must exist")
        
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                token_embd = nn.Embedding(config.vocab_size, config.n_embed),
                posit_embd = nn.Embedding(config.context_size, config.n_embed),
                dropout = nn.Dropout(config.drpout),
                blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = LayerNorm(config.n_embed, bias=config.bias)
            )
        )

        # Setting up map embeddings -> logits map 
        self.logits_map = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.transformer.token_embd.weight = self.logits_map.weight

        self.apply(self._init_weights)

        #for reproducing Karpathy, apply different wights init for projection layers
        for name, param in self.named_parameters():
            if name.endswith('proj.weight'):
                nn.init.normal_(param, mean=0.0, std = 0.2 / math.sqrt(2 * config.n_layer))


        print(f"Number of parameters = {self.get_num_params() * 1e-6}")


    def get_num_params(self, non_embeddings: bool = True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embeddings:
            n_params -= self.transformer.posit_embd.weight.numel()

        return n_params
    

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.2) #perhaps, I should rearrange it


    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor: 
        device = idx.device

        B, T = idx.size()
        if T > self.config.context_size:
            raise ValueError(f"Cannot forward sequence of length{T} with context_size of {self.config.context_size}")
        
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_emb = self.transformer.token_embd(idx)
        pos_emb = self.transformer.posit_embd(pos)

        X = self.transformer.dropout(tok_emb + pos_emb)

        for block in self.transformer.blocks:
            X = block(X)

        X = self.transformer.ln_f(X)


        if targets is not None:
            logits = self.logits_map(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        else: 
            logits = self.logits_map(X[:, [-1],:])
            loss = None


        return logits, loss
    
    def configure_optimization(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float | torch.Tensor, float | torch.Tensor], 
        device_type: str
    ) -> torch.optim.Optimizer:
        """
        This method tackles optimizer settings
        1. passes only trainable params to optimizer
        2. makes only matrix-like params decay while LayerNorms (cause sigmnal must scale propperly) and biases (cause why decay a set of independent shifts) do not decay
        For Regularization
        """
        all_train_params= [param for param in self.parameters() if param.requires_grad]
        
        decay_params = [param for param in all_train_params if param.dim() >= 2]
        nondecay_params = [param for param in all_train_params if param.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nondecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(param.numel() for param in decay_params)
        num_nondecay_params = sum(param.numel() for param in nondecay_params)
        print(f"Number of decayed parameters tensors {len(decay_params)} with total decayed parameter number {num_decay_params}")
        print(f"Number of decayed parameters tensors {len(nondecay_params)} with total decayed parameter number {num_nondecay_params}")

        fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused and device_type == 'cuda'
        if use_fused:
            print("fuse is used")
            extra_args = {
                'fused': use_fused
            }
        else:
            print("No fuse")
            extra_args = {}

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer
    


    @torch.no_grad()
    def generate(
        self, 
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.context_size else idx[:, -self.config.context_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx