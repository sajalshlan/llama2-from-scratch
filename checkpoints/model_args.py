# Checkpoint1 - ModelArg
# a blueprint for the entire model, kind of like a config file
# head_dim shouldn't be hardcoded — it's derived from dim // n_heads
# vocab_size not hardcoded so that it can be explictely set by tokenizer and model works with it

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096 # i was going to call it emb_dim cause it is more intuitive but dim is the standard used in llama
    n_layers: int = 32 # number of transformer blocks
    max_seq_length: int = 2048
    n_heads:int = 32
    n_kv_heads: Optional[int] = None # None = same as n_heads if multi-head attention, maybe 2 or 4 if grouped-query-attention 
    vocab_size: int = -1 # set from tokenizer
    ffn_dim_multiplier: Optional[float] = None # hidden layer dimension of ffn
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    device: Optional[str] = None
    multiple_of: int = 256 # this makes the FFN hidden dimension always a multiple of 256 for GPU memory efficiency
