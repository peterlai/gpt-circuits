# %% Imports (reuse from previous blocks)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Callable, Dict, Tuple
from jaxtyping import Float, Int
from torch import Tensor
from dataclasses import dataclass
import time # For timing comparison

# %% Device Setup (reuse)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}") # Already printed

# %% Model Definitions (reuse MLP, DummySAE)
@dataclass
class GPTConfig:
    n_embd: int
    bias: bool = False

class MLP(nn.Module): # (definition unchanged)
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.W_in = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.W_out = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, x):
        x = self.gelu(self.W_in(x))
        x = self.W_out(x)
        return x

class DummySAE(nn.Module): # (definition unchanged)
    def __init__(self, n_embd: int, n_features: int):
        super().__init__()
        self.n_features = n_features
        self.W_dec = nn.Parameter(torch.randn((n_features, n_embd), device=device) / n_features**0.5)
        self.W_enc = nn.Parameter(torch.randn((n_embd, n_features), device=device) / n_embd**0.5)
        self.b_enc = nn.Parameter(torch.zeros(n_features, device=device))
        self.b_dec = nn.Parameter(torch.zeros(n_embd, device=device))

    def encode(self, x: Float[Tensor, "... n_embd"]) -> Float[Tensor, "... n_features"]:
        return x @ self.W_enc + self.b_enc

    def decode(self, x: Float[Tensor, "... n_features"]) -> Float[Tensor, "... n_embd"]:
        return x @ self.W_dec + self.b_dec

    def decode_sparse(self,
                      feat_values: Float[Tensor, "batch seq k"],
                      feat_indices: Int[Tensor, "batch seq k"]
                      ) -> Float[Tensor, "batch seq n_embd"]:
        b, s, k = feat_indices.shape
        n_embd = self.W_dec.shape[1]
        W_dec_active = self.W_dec[feat_indices] # Shape: (b, s, k, n_embd)
        decoded = einops.einsum(feat_values, W_dec_active, "b s k, b s k e -> b s e")
        decoded += self.b_dec
        return decoded

# %% Modified Sandwich Function (reuse sandwich_sparse_input)
activations_store: Dict[str, Tensor] = {} # (definition unchanged)

def sandwich_sparse_input(
    in_feat_values : Float[Tensor, "batch seq k_in"],
    in_feat_indices: Int[Tensor, "batch seq k_in"],
    sae_mlpin : DummySAE, mlp : MLP, sae_mlpout : DummySAE,
    normalize : Callable[[Tensor], Tensor]
) -> Float[Tensor, "batch seq feat_out"]:
    normalized_values = normalize(in_feat_values)
    resid_mid = sae_mlpin.decode_sparse(normalized_values, in_feat_indices)
    mlp_post = mlp(resid_mid)
    resid_post = mlp_post + resid_mid
    resid_post_featmag = sae_mlpout.encode(resid_post)
    return resid_post_featmag

# %% Hook Function & Derivative Helper (reuse mlp_gelu_hook_fn, get_elementwise_derivative)
def mlp_gelu_hook_fn(module, inputs, outputs): # (definition unchanged)
    pre_actfn = inputs[0]
    pre_actfn_copy = pre_actfn.detach().requires_grad_(True)
    with torch.enable_grad():
        recomputed_post_actfn = F.gelu(pre_actfn_copy, approximate="tanh")
        grad_of_actfn = torch.autograd.grad(
            outputs=recomputed_post_actfn, inputs=pre_actfn_copy,
            grad_outputs=torch.ones_like(recomputed_post_actfn),
            retain_graph=False, create_graph=False
        )[0]
    activations_store["mlp_gelu_grad"] = grad_of_actfn
    return outputs

def get_elementwise_derivative(func: Callable, x: Tensor) -> Tensor: # (definition unchanged)
    x_detached = x.detach().requires_grad_(True)
    with torch.enable_grad():
        y = func(x_detached)
        grads = torch.autograd.grad(outputs=y, inputs=x_detached,
                                    grad_outputs=torch.ones_like(y),
                                    create_graph=False, retain_graph=False)[0]
    return grads

# %% Dense Analytical Jacobian (Optimized Version - reuse dense_jacobian_normalized_input_optimized)
def dense_jacobian_normalized_input_optimized(
    sae_mlpin : DummySAE, sae_mlpout : DummySAE, mlp: MLP,
    mlp_act_grads: Float[Tensor, "batch seq d_mlp"],
    normalize_func: Callable[[Tensor], Tensor],
    in_feat_mags: Float[Tensor, "batch seq feat_in"],
) -> Float[Tensor, "batch seq feat_out feat_in"]: # (definition unchanged)

    norm_grads = get_elementwise_derivative(normalize_func, in_feat_mags)
    b, s, feat_in = norm_grads.shape
    n_embd = mlp.W_in.weight.shape[1]
    d_mlp = mlp.W_in.weight.shape[0]
    feat_out = sae_mlpout.W_enc.shape[1]

    W_dec_in = sae_mlpin.W_dec
    W_enc_out = sae_mlpout.W_enc
    W_in = mlp.W_in.weight
    W_out = mlp.W_out.weight
    dtype = W_dec_in.dtype
    device = W_dec_in.device

    # Removed keyword args specifying dimensions
    J_mlp_local = einops.einsum(
        W_out, mlp_act_grads.to(dtype), W_in,
        "e_out d, b s d, d e_in -> b s e_out e_in"
    )

    diag_indices = torch.arange(n_embd, device=device)
    J_mlp_plus_I = J_mlp_local.clone() # Use clone to be safe
    J_mlp_plus_I[:, :, diag_indices, diag_indices] += 1.0

    # Removed keyword args specifying dimensions
    jacobian_prev = einops.einsum(
        W_enc_out.T, J_mlp_plus_I, W_dec_in.T,
        "f e_out, b s e_out e_in, e_in i -> b s f i"
    )

    total_jacobian = jacobian_prev * norm_grads.unsqueeze(-2)
    return total_jacobian

# %% Sparse Analytical Jacobian Calculation Function (Corrected)

def sparse_jacobian_normalized_input(
    sae_mlpin : DummySAE,
    sae_mlpout : DummySAE,
    mlp: MLP,
    mlp_act_grads: Float[Tensor, "batch seq d_mlp"],     # g' (dense, from hook)
    normalize_func: Callable[[Tensor], Tensor],
    in_feat_values: Float[Tensor, "batch seq k_in"],     # Sparse input values
    in_feat_indices: Int[Tensor, "batch seq k_in"],      # Sparse input indices
    out_feat_indices: Int[Tensor, "batch seq k_out"],    # Target sparse output indices
) -> Float[Tensor, "batch seq k_out k_in"]:              # Jacobian for sparse elements
    """
    Computes the analytical Jacobian d(out_features[out_indices])/d(in_features[in_indices])
    efficiently using sparse indices. (Corrected: Removed unnecessary einsum kwargs)
    """
    # --- 1. Calculate Derivative of Normalization for *Active* Inputs ---
    norm_grads_sparse = get_elementwise_derivative(normalize_func, in_feat_values)
    # Shape: (batch, seq, k_in)
    b, s, k_in = in_feat_indices.shape
    k_out = out_feat_indices.shape[-1]
    n_embd = mlp.W_in.weight.shape[1]
    d_mlp = mlp.W_in.weight.shape[0]
    dtype = mlp.W_in.weight.dtype
    device = mlp.W_in.weight.device

    # --- 2. Get Sparse Weights ---
    W_dec_in_active = sae_mlpin.W_dec[in_feat_indices]        # (b, s, k_in, n_embd)
    W_enc_out_T_active = sae_mlpout.W_enc.T[out_feat_indices] # (b, s, k_out, n_embd)
    W_in = mlp.W_in.weight                                   # (d_mlp, n_embd)
    W_out = mlp.W_out.weight                                 # (n_embd, d_mlp)

    # --- 3. Calculate the Inner MLP Jacobian Term (J_mlp_local + I) ---
    # Removed keyword args specifying dimensions
    J_mlp_local = einops.einsum(
        W_out, mlp_act_grads.to(dtype), W_in,
        "e_out d, b s d, d e_in -> b s e_out e_in"
    )
    # Shape: (b, s, n_embd, n_embd)

    diag_indices = torch.arange(n_embd, device=device)
    J_mlp_plus_I = J_mlp_local # Modify in place (or clone if needed)
    J_mlp_plus_I[:, :, diag_indices, diag_indices] += 1.0
    # Shape: (b, s, n_embd, n_embd)

    # --- 4. Perform Sparse Outer Multiplications ---
    W_dec_in_active_T = W_dec_in_active.transpose(-2, -1) # (b, s, n_embd, k_in)

    # Removed keyword args specifying dimensions (they weren't here before, but confirming)
    jacobian_prev_sparse = einops.einsum(
        W_enc_out_T_active,         # (b, s, k_out, n_embd)
        J_mlp_plus_I,               # (b, s, n_embd, n_embd)
        W_dec_in_active_T,          # (b, s, n_embd, k_in)
        "b s ko e_out, b s e_out e_in, b s e_in ki -> b s ko ki"
    )
    # Shape: (b, s, k_out, k_in)

    # --- 5. Apply Sparse Normalization Derivative ---
    total_jacobian_sparse = jacobian_prev_sparse * norm_grads_sparse.unsqueeze(-2)
    # Shape: (b, s, k_out, k_in)

    return total_jacobian_sparse


# %% Main Execution Block (Identical to previous, just uses corrected function)
# %% Main Execution Block (with Intermediate Checks)
if __name__ == "__main__":
    # --- Hyperparameters ---
    # Use the dimensions from the error report for consistency
    n_embd = 32      # Adjusted based on typical d_mlp = 4*n_embd and previous examples
    n_feat = 160     # From dense jacobian shape
    k_in = 10       # From sparse jacobian shape
    k_out = 10      # From sparse jacobian shape
    batch_size = 3  # From sparse jacobian shape
    seq_len = 7     # From sparse jacobian shape
    d_mlp = 4 * n_embd # Make sure this aligns if possible, or use a known d_mlp

    print(f"Using dimensions: B={batch_size}, S={seq_len}, N_feat={n_feat}, K_in={k_in}, K_out={k_out}, N_embd={n_embd}")

    # --- Instantiate Models ---
    # Ensure consistent dimensions
    if d_mlp != 4 * n_embd:
      print(f"Warning: d_mlp ({d_mlp}) not 4*n_embd ({4*n_embd}). Adjusting n_embd based on d_mlp from hook if needed.")
      # This shouldn't happen if mlp_gelu_grads shape is consistent, but good check.

    gpt_config = GPTConfig(n_embd=n_embd)
    mlp = MLP(gpt_config).to(device)
    # Ensure SAEs use n_feat
    sae_mlpin = DummySAE(n_embd, n_feat).to(device)
    sae_mlpout = DummySAE(n_embd, n_feat).to(device)


    # --- Generate Sparse Input Data ---
    in_feat_indices = torch.randint(0, n_feat, (batch_size, seq_len, k_in), device=device)
    # Make indices unique *within* the last dimension (k_in) for realistic sparse vectors
    # This is tricky to guarantee globally unique across batch/seq easily, focus on k_in uniqueness
    for b in range(batch_size):
        for s in range(seq_len):
            unique_indices = torch.unique(in_feat_indices[b, s])
            if len(unique_indices) < k_in:
                # Simple fix: just take the first k_in unique if available
                 if len(unique_indices) >= k_in:
                     in_feat_indices[b, s] = unique_indices[:k_in]
                 else: # If not enough unique, pad with duplicates (less ideal but works)
                     in_feat_indices[b, s, :len(unique_indices)] = unique_indices
                     in_feat_indices[b, s, len(unique_indices):] = unique_indices[0] # Pad with first
            else:
                 in_feat_indices[b, s] = unique_indices[:k_in] # Ensure exactly k_in

    in_feat_values = torch.randn(batch_size, seq_len, k_in, device=device)

    # --- Define Normalization Function ---
    def normalize_fn(x):
        return F.gelu(x, approximate="tanh")

    # --- Run Forward Pass with Hook using Sparse Input ---
    activations_store.clear()
    hook_handle = mlp.gelu.register_forward_hook(mlp_gelu_hook_fn)
    dense_output_mags = sandwich_sparse_input(
        in_feat_values, in_feat_indices,
        sae_mlpin, mlp, sae_mlpout, normalize_fn
    )
    hook_handle.remove()
    if "mlp_gelu_grad" not in activations_store:
        raise RuntimeError("Hook did not capture MLP GELU gradients.")
    mlp_gelu_grads = activations_store["mlp_gelu_grad"]
    d_mlp_actual = mlp_gelu_grads.shape[-1]
    if d_mlp != d_mlp_actual:
        print(f"Warning: Config d_mlp {d_mlp} != actual d_mlp from hook {d_mlp_actual}. Using {d_mlp_actual}.")
        d_mlp = d_mlp_actual


    # --- Select Target Output Indices ---
    _, out_feat_indices = torch.topk(dense_output_mags.abs(), k=k_out, dim=-1)

    # --- Intermediate Calculation: J_mlp_plus_I (used by both) ---
    # Calculate this once based on the hook gradients
    W_in = mlp.W_in.weight           # (d_mlp, n_embd)
    W_out = mlp.W_out.weight         # (n_embd, d_mlp)
    dtype = mlp.W_in.weight.dtype
    device = mlp.W_in.weight.device

    J_mlp_local = einops.einsum(
        W_out, mlp_gelu_grads.to(dtype), W_in,
        "e_out d, b s d, d e_in -> b s e_out e_in"
    )
    diag_indices = torch.arange(n_embd, device=device)
    J_mlp_plus_I = J_mlp_local.clone() # Clone to avoid modifying if reused
    J_mlp_plus_I[:, :, diag_indices, diag_indices] += 1.0
    print(f"Intermediate J_mlp_plus_I shape: {J_mlp_plus_I.shape}") # Should be (b, s, n_embd, n_embd)

    # --- Intermediate Calculation: jacobian_prev_sparse ---
    print("\nCalculating jacobian_prev_sparse (before norm grad scaling)...")
    W_dec_in_active = sae_mlpin.W_dec[in_feat_indices]        # (b, s, k_in, n_embd)
    W_enc_out_T_active = sae_mlpout.W_enc.T[out_feat_indices] # (b, s, k_out, n_embd)
    W_dec_in_active_T = W_dec_in_active.transpose(-2, -1)      # (b, s, n_embd, k_in)

    jacobian_prev_sparse = einops.einsum(
        W_enc_out_T_active, J_mlp_plus_I, W_dec_in_active_T,
        "b s ko e_out, b s e_out e_in, b s e_in ki -> b s ko ki"
    )
    print(f"jacobian_prev_sparse shape: {jacobian_prev_sparse.shape}")

    # --- Intermediate Calculation: jacobian_prev_dense ---
    print("\nCalculating jacobian_prev_dense (before norm grad scaling)...")
    W_dec_in = sae_mlpin.W_dec       # (n_feat, n_embd)
    W_enc_out = sae_mlpout.W_enc     # (n_embd, n_feat)

    jacobian_prev_dense = einops.einsum(
        W_enc_out.T, J_mlp_plus_I, W_dec_in.T,
        "f e_out, b s e_out e_in, e_in i -> b s f i"
    )
    print(f"jacobian_prev_dense shape: {jacobian_prev_dense.shape}")

    # --- Intermediate Selection: jacobian_prev_dense_selected ---
    print("\nSelecting sub-matrix from jacobian_prev_dense...")
    b_idx = torch.arange(batch_size, device=device)[:, None, None, None].expand(-1, seq_len, k_out, k_in)
    s_idx = torch.arange(seq_len, device=device)[None, :, None, None].expand(batch_size, -1, k_out, k_in)
    out_idx_expanded = out_feat_indices.unsqueeze(-1).expand(-1, -1, -1, k_in) # (b, s, k_out, k_in)
    in_idx_expanded = in_feat_indices.unsqueeze(-2).expand(-1, -1, k_out, -1)   # (b, s, k_out, k_in)

    jacobian_prev_dense_selected = jacobian_prev_dense[b_idx, s_idx, out_idx_expanded, in_idx_expanded]
    print(f"jacobian_prev_dense_selected shape: {jacobian_prev_dense_selected.shape}")

    # --- Intermediate Comparison 1: Pre-Normalization Jacobians ---
    print("\nComparing PRE-NORMALIZATION Jacobians (Sparse vs Selected Dense)...")
    try:
        torch.testing.assert_close(jacobian_prev_sparse, jacobian_prev_dense_selected, rtol=1e-5, atol=1e-6) # Tighter tolerance
        print("✅ Pre-Normalization Jacobians match!")
        pre_norm_match = True
    except AssertionError as e:
        print("❌ Pre-Normalization Jacobians DO NOT match! Error is likely in sparse weight gathering or einsum (Steps 2-4).")
        print(e)
        pre_norm_match = False


    # --- Intermediate Check 2: Normalization Gradients ---
    print("\nComparing Normalization Gradients (Sparse vs Selected Dense)...")
    norm_grads_sparse = get_elementwise_derivative(normalize_fn, in_feat_values) # (b, s, k_in)

    # Create dense input for comparison grad calculation
    in_feat_mags_dense = torch.zeros(batch_size, seq_len, n_feat, device=device, dtype=in_feat_values.dtype)
    in_feat_mags_dense.scatter_(dim=-1, index=in_feat_indices, src=in_feat_values)
    norm_grads_dense = get_elementwise_derivative(normalize_fn, in_feat_mags_dense) # (b, s, n_feat)

    # Select gradients corresponding to sparse indices
    # Use gather for potentially cleaner selection than the previous indexing attempt
    norm_grads_dense_selected = torch.gather(norm_grads_dense, dim=-1, index=in_feat_indices) # (b, s, k_in)

    try:
        torch.testing.assert_close(norm_grads_sparse, norm_grads_dense_selected, rtol=1e-6, atol=1e-7) # Very tight tolerance
        print("✅ Normalization Gradients match for active indices!")
        norm_grad_match = True
    except AssertionError as e:
        print("❌ Normalization Gradients DO NOT match! Check dense input creation or derivative fn.")
        print(e)
        norm_grad_match = False

    # --- Calculate Final Jacobians (using already computed parts if possible) ---
    print("\nCalculating Final Jacobians...")
    start_time_sparse = time.time()
    total_jacobian_sparse = jacobian_prev_sparse * norm_grads_sparse.unsqueeze(-2)
    sparse_time = time.time() - start_time_sparse # Time only the final step now
    print(f"Final Sparse Jacobian calculation time: {sparse_time:.6f} seconds")

    start_time_dense = time.time()
    total_jacobian_dense_full = jacobian_prev_dense * norm_grads_dense.unsqueeze(-2)
    # Selection step
    total_jacobian_dense_selected = total_jacobian_dense_full[b_idx, s_idx, out_idx_expanded, in_idx_expanded]
    dense_time = time.time() - start_time_dense # Time final step + selection
    print(f"Final Dense Jacobian calculation + Selection time: {dense_time:.6f} seconds")

    # --- Final Comparison ---
    print("\nComparing FINAL Jacobians (Sparse vs Selected Dense)...")
    if not pre_norm_match:
        print("Skipping final comparison because Pre-Normalization Jacobians mismatched.")
    elif not norm_grad_match:
         print("Skipping final comparison because Normalization Gradients mismatched.")
    else:
        try:
            # Use the original tolerances from the error report
            torch.testing.assert_close(total_jacobian_sparse, total_jacobian_dense_selected, rtol=1e-4, atol=1e-5)
            print("✅ FINAL Jacobians match!")
        except AssertionError as e:
            print("❌ FINAL Jacobians DO NOT match! Discrepancy likely arises in final scaling or subtle FP differences amplified.")
            print(e)
            diff = (total_jacobian_sparse - total_jacobian_dense_selected).abs()
            print(f"Max difference: {diff.max().item()}")
            print(f"Mean difference: {diff.mean().item()}")
# %%