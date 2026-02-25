# Checkpoint2 - RMS Normalization

# why normalise at all?
# 3 reasons - 
#    1. the inputs shouldn't be too large or small that multiplications across layers starts exploding/vanishing making training unstable(var=1 should have)
#    2. inputs shouldn't be too skewed so that activation functions saturate(mean=0 should have)
#    3. inputs shouln't keep shifting their distribution every step (covariate shift) [Normalization standardizes what each layer receives, so it always sees inputs from roughly the same distribution (mean 0, variance 1), regardless of what happened upstream. Each layer can now focus on learning its own transformation, rather than constantly chasing a moving target.]

# standard transformer uses LayerNorm, LLaMA2 uses RMS Norm. difference?
# LayerNorm(GPT-2) normalizes by subtracting the mean and dividing the standard deviation
# RMSNorm(LLaMA) says the mean-centering is expensive and doesn't help much, just divide by the size of the vector
#   - just one learnable parameter(gamma), no mean subtraction, no bias param


# we get vector in input (tensor here) -> we normalise it and return the tensor back
# inputs(x) will be of the dimension - (batch, seq_len, dim)
# need a learnable parameter, and the formula for rms normalization in init
# formula - [x/RMS(x)].Gamma  -> RMS(x) = (standard deviation) underroot of squared summation of terms in the input tensor divided by number of terms

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps: float=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim)) # because we work at dim level, given an input x looks like this - (batch, seq_len, dim)
                                                   # nn.Parameter tells pyTorch - this tensor should be updated during backprop
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms_norm = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).rsqrt() # square every element, then mean over last dimension, and then 1 by square root.
        return x*rms_norm*self.gamma
        



