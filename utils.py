from pathlib import Path
from typing import Callable
import torch
import yaml

from model import ModelConfig
from train import TrainConfig



def load_config(path: str | Path):
    """Load ``train`` and ``model`` sections from a YAML file.

    The model section is converted to :class:`model.ModelConfig`; the train
    section is converted to :class:`TrainConfig`. GPT-2 loading options are
    intentionally not part of this configuration surface.
    """

    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("config YAML must contain a mapping")

    train_values = raw.get("train", {})
    model_values = raw.get("model", {})
    if not isinstance(train_values, dict) or not isinstance(model_values, dict):
        raise ValueError("'train' and 'model' sections must be mappings")

    # Keep the format invariant in code, rather than allowing a future
    # checkpoint backend to be selected accidentally from YAML.
    train_values = dict(train_values)
    train_values["checkpoint_format"] = "safetensors"
    return TrainConfig(**train_values), ModelConfig(**model_values)


def load_text(path: str) -> str:
    with open(path, 'r') as t:
        text = t.read()

    print("length of the dataset in characters = ", len(text))
    return text


def get_vocabulary(text: str) -> tuple[dict[int,str], dict[str, int], Callable, Callable]:
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print("vocabulary size = ", vocab_size)

    stoi = {ch: i for i,ch in enumerate(chars)}
    itos = {i: ch for i,ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    return itos, stoi, encode, decode

def get_train_val_split(test_ratio: float, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, encode, _ = get_vocabulary(text)
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(test_ratio * len(data))
    train = data[:n]
    val = data[n:]
    return train, val


def get_batch(data: torch.Tensor, context_size: int, batch_size: int, device: str = 'cpu'):
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i:i+context_size] for i in ix])
    y = torch.stack([data[i+1:i+context_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model: torch.Module, 
    train: torch.Tensor, 
    val: torch.Tensor, 
    eval_iters: int,
    batch_size: int,
    context_size: int,
    device: str,
)-> dict:
    out = {}
    model.eval()
    splits = {'train': train, 'val': val}
    for split, data in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(data, context_size, batch_size, device)
            _, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out
