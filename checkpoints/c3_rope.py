# Checkpoint3 - RoPE Embeddings

# The Implementation Has Two Steps

# Step 1: Precompute the rotation frequencies (done once, not per forward pass)
# For each position 0 to max_seq_len
# For each dimension pair 0 to head_dim/2
# Compute the complex number e^(iθ) = cos(θ) + i·sin(θ)

# Step 2: Apply the rotation (done every forward pass, to Q and K)
# Treat pairs of dimensions as complex numbers
# Multiply by the precomputed frequencies (complex multiplication = rotation)

