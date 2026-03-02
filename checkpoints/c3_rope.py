# Checkpoint3 - RoPE Embeddings

# The Implementation Has Two Steps

# Step 1: Precompute the rotation frequencies (done once, not per forward pass) - This function runs once at startup. It precalculates all the rotation angles for every position and every dimension pair. theta=10000 is a constant from the original RoPE paper.
    # For each position 0 to max_seq_len
    # For each dimension pair 0 to head_dim/2
    # Compute the complex number e^(iθ) = cos(θ) + i·sin(θ)

# Step 2: Apply the rotation (done every forward pass, to Q and K)
    # Treat pairs of dimensions as complex numbers
    # Multiply by the precomputed frequencies (complex multiplication = rotation)

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arrange(0, head_dim, 2).float() / head_dim))
    positions = torch.outer(positions.float(), freqs)
    angles = torch.outer(positions.float(), freqs)
    freqs_complex = torch.polar(torch.ones_like(angles), angles) # converts each angle into a complex number on the unit circle: `cos(angle) + i·sin(angle)`. complex multiplication is rotation.
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    rotated = x_complex * freqs_complex
    out = torch.view_as_real(rotated).reshape(*x.shape)
    return out.type_as(x).to(device)
