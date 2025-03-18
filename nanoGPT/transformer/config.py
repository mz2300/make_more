from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_blocks: int
    block_size: int
    vocab_size: int 
    n_heads: int
    n_embd: int
    dropout: float = 0.0
    device: str = 'cpu'


config_args = { 
    'vocab_size': None,
    'n_blocks' : 4,
    'block_size': 128,
    'n_heads': 8,
    'n_embd': 192,
    'dropout': 0.2
}

config_default = ModelConfig(**config_args)