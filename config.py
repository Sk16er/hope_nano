"""
HOPE Model Configuration
"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class HOPEConfig:
    vocab_size: int = 50257 # GPT-2 vocab size
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    block_size: int = 256
    dropout: float = 0.1
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    
    # HOPE specific
    cms_update_periods: List[int] = field(default_factory=lambda: [1, 4, 16]) # Update periods for CMS blocks
    learning_rate_memory: float = 1e-2 # Learning rate for the inner loop (Titans/CMS)
    
    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
