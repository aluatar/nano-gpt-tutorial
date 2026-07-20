# Training configuration

Training and model settings are defined in YAML and loaded with
`utils.load_config`. The loader returns a `(TrainConfig, ModelConfig)` pair.
This repository uses character-level tokenization and does not load GPT-2
checkpoints.

## YAML format

Use separate `train` and `model` mappings:

```yaml
train:
  learning_rate: 0.0003
  checkpoint_path: checkpoints
  dataset_path: data/Pushkin.txt
  tokenizer_method: character

model:
  vocab_size: 144
  context_size: 124
  n_layer: 6
  n_head: 8
  n_embed: 256
  drpout: 0.25
  bias: true
```

`checkpoint_format` is always set to `safetensors` by the loader. It is not a
user-selectable YAML option, which prevents accidentally introducing another
checkpoint format.

## Loading configuration

```python
from utils import load_config

train_config, model_config = load_config("config/train.yaml")
print(train_config.dataset_path)
print(model_config.context_size)
```

`load_config(path)` accepts a string or `pathlib.Path`. It uses
`yaml.safe_load`, requires the YAML document to be a mapping, and converts the
two sections into their respective dataclasses. Missing sections use empty
mappings, so `model.vocab_size` remains the only required model field.

## `TrainConfig`

`TrainConfig` contains:

- `learning_rate`: positive AdamW learning rate.
- `checkpoint_path`: directory or path used by the training loop for saved
  checkpoints.
- `dataset_path`: input text file.
- `tokenizer_method`: currently only `character` is accepted.
- `checkpoint_format`: fixed to `safetensors`.

Construction validates the learning rate, tokenizer, and checkpoint format and
raises `ValueError` for unsupported values.

## `ModelConfig`

`ModelConfig` is defined in `model.py` and controls the Transformer:

- `vocab_size`: number of character tokens (required).
- `context_size`: maximum sequence length.
- `n_layer`: number of Transformer blocks.
- `n_head`: attention heads per block.
- `n_embed`: embedding width; it must be divisible by `n_head`.
- `drpout`: dropout probability (the spelling matches the current model API).
- `bias`: whether linear and layer-normalization bias parameters are used.

Example model construction:

```python
from model import Transformer

model = Transformer(model_config)
```

The current training entry point is still being assembled; this document
describes the configuration API and its invariants, not a completed checkpoint
save/resume workflow.

## Data helpers in `utils.py`

- `load_text(path)` reads a UTF-8 text dataset and reports its character count.
- `get_vocabulary(text)` builds sorted character-to-index and index-to-character
  mappings and returns `encode`/`decode` callables.
- `get_train_val_split(test_ratio, text)` encodes the text and returns the
  sequential training and validation tensors. Despite its historical name,
  `test_ratio` is the fraction assigned to training data.
- `get_batch(data, context_size, batch_size, device)` samples random contiguous
  input windows and next-token targets, returning tensors shaped
  `(batch_size, context_size)`.
- `estimate_loss(model, train, val, eval_iters, batch_size, context_size,
  device)` evaluates both splits without gradients and returns their mean loss.
