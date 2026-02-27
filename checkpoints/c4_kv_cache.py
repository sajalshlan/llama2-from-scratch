# Checkpoint 4 - KV Cache

# Keep storing the key and value tensors from all the previous tokens. Just calculate the new ones and keep appending them to the cache so that we don't have to do the same computation again and again.

# One entry per token, per layer, per KV Head
# input shape - (batch, seq_len_so_far, n_kv_heads, head_dim)
# but we preallocate the max size upfront(faster than dynamic resizing)

# some Q/A claude asked me and my answers - 
# Q: Tensors don't have .append() — how do you insert xk into a specific position in a preallocated tensor?
# A: self.cache_k[:batch_size, start_pos] = xk

# Q: cache_k and cache_v shouldn't be nn.Parameters (not learned). But if you just do self.cache_k = torch.zeros(...), PyTorch won't move them to GPU when you call model.to(device). What's the fix?
# A: self.cache_k = nn.Parameter(..., requires_grad=False)

    # now claude's answers
        # Q1: Correct! [:batch_size, start_pos] — exactly right. You're writing into the right batch slice, at the right position.

        # Q2: Close, but register_buffer is the cleaner way. Here's why:
        # nn.Parameter(..., requires_grad=False) technically works but it's semantically wrong — Parameters are meant for learned weights. Using it for a cache is confusing and it'll show up in model.parameters() which could cause issues with optimizers.
        # register_buffer is PyTorch's explicit way of saying "this tensor belongs to the module, move it with the model, but don't train it."

import torch
import torch.nn as nn

class KVCache(nn.Module):
    def __init__(self, max_batch_size: int, max_seq_len: int, n_kv_heads: int, head_dim: int, device: str):
        super().__init__()
        self.register_buffer('cache_k', torch.zeros(max_batch_size, max_seq_len, n_kv_heads, head_dim))
        self.register_buffer('cache_v', torch.zeros(max_batch_size, max_seq_len, n_kv_heads, head_dim))


    def update(self, batch_size: int, start_pos: int, xk: torch.Tensor, xv: torch.Tensor):
        self.cache_k[:batch_size, start_pos] = xk
        self.cache_v[:batch_size, start_pos] = xv
        return self.cache_k[:batch_size, : start_pos+1], self.cache_v[:batch_size, : start_pos+1]