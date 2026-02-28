# Checkpoint 6 - FeedForward(SwiGLU)

# FFN(x) = W2( Swish(W1·x) ⊗ W3·x ) 

# So conceptually, what's happening is that after attention, each token has generated information from other tokens. This feed-forward neural network processes that information for each token independently, so there is no interaction between tokens. The model has gathered whatever context it needed; now it will do the main thinking part, the main computation part, which the feed-forward network is responsible for.

# Standard FFN vs SwiGLU
    # Standard FFN (GPT-2):
    # x → Linear → ReLU → Linear → output

    # SwiGLU (LLaMA):
    # x → Linear (W1) → Swish ⊗ Linear (W3) → Linear (W2) → output

# Two key differences:
    # Three linear layers instead of two
    # Swish gate — instead of just activating, you multiply by a learned gate

# The formula:
    # FFN(x) = W2( Swish(W1·x) ⊗ W3·x )
    # Where ⊗ is elementwise multiplication and Swish(x) = x · sigmoid(x). In PyTorch this is just F.silu(x) (SiLU and Swish are the same thing).


# Hidden dimension is scaled down by 2/3 -> [ hidden_dim = 4 * dim * (2/3) ]
    # LLaMA uses SwiGLU which has three weight matrices instead of two. This makes the FFN ~50% more expensive than standard. To compensate and keep the total parameter count similar, they scale the hidden dim down by 2/3. So you get the benefits of SwiGLU without blowing up the parameter count.

# Why Round to multiple_of?
    # GPUs are most efficient when tensor dimensions are multiples of certain numbers (64, 128, 256). If your hidden dim is 1365, the GPU wastes cycles on padding. Rounding up to the nearest 256 keeps everything GPU-friendly.

# ReLU is a blunt instrument — it just zeros out negatives. The model has no control over *which* information survives.

# SwiGLU adds a **learned gate**:
    # signal = Swish(W1·x)   ← transformed signal
    # gate   = W3·x          ← learned gate (what to let through)
    # output = W2(signal ⊗ gate)
# `signal ⊗ gate` is elementwise multiply — each dimension of the signal gets scaled by the corresponding gate value. The gate learns "dimension 42 is important here, amplify it. Dimension 107 is noise, suppress it."

import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # compute hidden_dim (use the formula above)
        # define w1, w2, w3 as Linear layers (no bias)
        self.hidden_dim = int(4 * args.dim * 2/3) if args.ffn_dim_multiplier is None else args.ffn_dim_multiplier * args.dim
        self.hidden_dim = args.multiple_of * ((self.hidden_dim + args.multiple_of - 1) // args.multiple_of) # making it a multiple of gpu efficiency
        self.w1 = nn.Linear(args.dim, self.hidden_dim, bias=False)  # dim → hidden
        self.w2 = nn.Linear(self.hidden_dim, args.dim, bias=False)  # hidden → dim (compress back)
        self.w3 = nn.Linear(args.dim, self.hidden_dim, bias=False)  # dim → hidden (the gate)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # implement: W2( SiLU(W1·x) * W3·x )
        return self.w2(F.silu(self.w1(x)) * self.w3(x)))
        #F.silu(self.w1(x))          ← Swish(W1·x) — the signal
        #* self.w3(x)                ← ⊗ W3·x — the gate
        #self.w2(...)                ← W2( ... ) — compress back to dim
