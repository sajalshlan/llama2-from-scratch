# Checkpoint 7 - Decoder Block
# Attention + FFN stitched together with RMSNorm and residual connections.

# x → RMSNorm → SelfAttention → + residual → RMSNorm → FeedForward → + residual

# Why residual connections? 
    # So gradients can flow directly back through the network without passing through attention or FFN. Prevents vanishing gradients across 32 layers.

# Why norm before attention (Pre-Norm)?
#


# Why two separate RMSNorm instances? Each has its own gamma parameter that gets trained independently. If you used the same instance for both, they'd share weights — the norm before attention would interfere with the norm before FFN.


import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        self.norm1 = RMSNorm(args.dim)   # before attention
        self.norm2 = RMSNorm(args.dim)   # before FFN

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # apply the pattern:
        # x = x + attention(norm(x))
        # x = x + ffn(norm(x))
        x = x + self.self_attention(self.rms_norm1(x), start_pos, freqs_complex)
        x = x + self.feed_forward(self.rms_norm2(x))
        return x