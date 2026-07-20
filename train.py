from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainConfig:
    """Training and data settings loaded from a YAML configuration."""

    learning_rate: float = 3e-4
    checkpoint_path: str = "checkpoints"
    dataset_path: str = "data/Pushkin.txt"
    tokenizer_method: str = "character"
    checkpoint_format: str = "safetensors"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.tokenizer_method != "character":
            raise ValueError("only the 'character' tokenizer is supported")
        if self.checkpoint_format != "safetensors":
            raise ValueError("checkpoints must use the 'safetensors' format")