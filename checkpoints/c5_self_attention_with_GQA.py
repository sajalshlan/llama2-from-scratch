# Checkpoint 5 - Self Attention with GQA

# attention_score = softmax(Q·K^T / sqrt(head_dim)) · V
# Why divide by sqrt(head_dim)? Because dot products grow large when head_dim is large, pushing softmax into saturation (all weight on one token). Dividing stabilizes it.

# Multi-Head (MHA): Every head has its own Q, K, V projections. 8 heads = 8 sets of Q, K, V.
# Grouped Query Attention (GQA): Every head has its own Q, but K and V are shared across groups. Why? K and V are what gets cached — fewer KV heads = much smaller KV cache = less memory.

# The Full Forward Pass Flow

#   input x (batch, seq_len, dim)
#          ↓
#   Linear projections → xq, xk, xv
#          ↓
#   Reshape into heads
#          ↓
#   Apply RoPE to xq and xk
#          ↓
#   Update KV cache, get all keys/values so far
#          ↓
#   Repeat KV heads to match Q heads (GQA)
#          ↓
#   Compute attention scores: Q·K^T / sqrt(head_dim)
#          ↓
#   Apply causal mask (can't attend to future tokens)
#          ↓
#   Softmax → weighted sum with V
#          ↓
#   Reshape and project output

# this is how KV is trained for Q - repeat_interleave respects these trained relationships. repeat scrambles them.
# repeat_interleave:  .K0, K0, K0, K0, K1, K1, K1, K1  ← each element repeated
# repeat:             .K0, K1, K0, K1, K0, K1, K0, K1  ← whole sequence repeated

# seq_len = the number of tokens in the current input being processed. During inference this is always 1 — you're feeding one new token at a time.
# seq_so_far = all tokens processed up to this point, including everything in the KV cache.
    # seq_len    = 1    ← just the new token
    # seq_so_far = 51   ← all tokens Q needs to attend over

import torch
import math
import torch.nn as nn
import KVCache from c4_kv_cache


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # define:
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        self.n_heads, self.n_rep = args.n_heads, args.n_heads//self.n_kv_heads
        self.head_dim = args.dim//self.n_heads
        self.wq = nn.Linear(args.dim, self.n_heads*self.head_dim, bias=False)   # PyTorch loops over batch and seq_len automatically so no need to specify them
        self.wk = nn.Linear(args.dim, self.n_kv_heads*self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads*self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads*self.head_dim, args.dim, bias=False)
        self.cache = KVCache(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim, args.device)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # x shape: (batch, 1, dim)  ← during inference, one token at a time
        
        # steps:
        # 1. project x → xq, xk, xv
        batch, seq_len, _ = x.shape
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # 2. reshape into heads
        xq = xq.view(batch, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # 3. apply RoPE (call apply_rotary_embeddings)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # 4. update KV cache
        keys, values = self.cache.update(batch, start_pos, xk, xv)

        # 5. repeat KV heads to match Q heads
        keys = torch.repeat_interleave(keys, repeats=self.n_heads//self.n_kv_heads, dim=2)
        values = torch.repeat_interleave(values, repeats=self.n_heads//self.n_kv_heads, dim=2)

        # 6. compute attention: scores = Q·K^T / sqrt(head_dim)
        # 7. apply softmax

        xq = xq.transpose(1,2)  # (batch, n_heads, seq_len, head_dim)
        keys = keys.transpose(1,2)  # (batch, n_heads, seq_so_far, head_dim)
        values = values.transpose(1,2) # (batch, n_heads, seq_so_far, head_dim)
        scores = xq @ keys.transpose(2,3) # (batch, n_heads, seq_len, seq_so_far)
        scores = scores/math.sqrt(self.head_dim)
        scores = torch.softmax(scores, dim=-1)

        # 8. weighted sum with V
        # 9. reshape and project output
        output = scores @ values # (batch, n_heads, seq_len, head_dim)
        output = output.transpose(1, 2) # (batch, seq_len, n_heads, head_dim)
        output = output.contiguous().view(batch, seq_len, self.n_heads * self.head_dim) #After .transpose(), the tensor's memory layout becomes non-contiguous. .view() requires contiguous memory, so you need to call .contiguous() first to fix the layout.
        output = self.wo(output) # (batch, seq_len, dim)





        