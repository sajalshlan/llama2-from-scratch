# Checkpoint 8 - Transformer block

# input token ids (batch, seq_len)
#         ↓
# Embedding lookup → (batch, seq_len, dim)
#         ↓
# Precompute RoPE frequencies
#         ↓
# Pass through N DecoderBlocks
#         ↓
# Final RMSNorm
#         ↓
# Linear projection → logits (batch, seq_len, vocab_size)


# new thing - embedding
    # self.embedding = nn.Embedding(vocab_size, dim)
    # nn.Embedding is just a lookup table — a matrix of shape (vocab_size, dim). Token ID 42 → row 42 of that matrix. It's a learnable parameter, trained alongside everything else.

import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.dim)
        self.final_rms_norm = RMSNorm(args.dim)
        self.layers = nn.ModuleList([DecoderBlock(args) for _ in range(args.n_layers)])
        self.output_layer = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.freqs_complex = precompute_theta_pos_frequencies(
            args.dim // args.n_heads, 
            args.max_seq_len, 
            device=args.device
        )

    def forward(self, x: torch.Tensor, start_pos: int):
        # x shape: (batch, seq_len) — token ids
        batch, seq_len = x.shape
        # steps:
        # 1. embed tokens
        x = self.embedding(x)

        # 2. get freqs_complex for current positions
        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]

        # 3. pass through all decoder blocks
        for layer in self.layers:
            x = layer(x, start_pos, freqs_complex)

        # 4. apply final norm
        x = self.final_rms_norm(x)

        # 5. project to vocab size
        output = self.output_layer(x)

        # 6. return logits
        return output
        