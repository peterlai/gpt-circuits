# %% Imports
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Callable, Dict
from jaxtyping import Float
from torch import Tensor
from dataclasses import dataclass

# %% Device Setup
# Set device (CPU or GPU if available)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Optional: Choose specific GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% Model Definitions

@dataclass
class GPTConfig:
    """Minimal config for MLP"""
    n_embd: int
    bias: bool = False # Bias is False for standard GPT-2 MLP

class MLP(nn.Module):
    """Standard GPT-2 MLP Layer"""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.W_in = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.W_out = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh") # Using tanh approximation like GPT-2

    def forward(self, x):
        x = self.gelu(self.W_in(x))
        x = self.W_out(x)
        return x

class DummySAE(nn.Module):
    """Simple SAE structure for demonstration"""
    def __init__(self, n_embd: int, n_features: int):
        super().__init__()
        # Note: nn.Linear weights are (d_out, d_in)
        # We define W_dec as (feat, embd) and W_enc as (embd, feat)
        # for consistent matrix multiplication notation (x @ W)
        self.W_dec = nn.Parameter(torch.randn((n_features, n_embd), device=device) / n_features**0.5)
        self.W_enc = nn.Parameter(torch.randn((n_embd, n_features), device=device) / n_embd**0.5)
        self.b_enc = nn.Parameter(torch.zeros(n_features, device=device))
        self.b_dec = nn.Parameter(torch.zeros(n_embd, device=device))


    def encode(self, x: Float[Tensor, "... n_embd"]) -> Float[Tensor, "... n_features"]:
        # Encode: embd -> features
        # y = x @ W_enc + b_enc (ReLU is often here, but omitted for Jacobian linearity)
        return x @ self.W_enc + self.b_enc

    def decode(self, x: Float[Tensor, "... n_features"]) -> Float[Tensor, "... n_embd"]:
        # Decode: features -> embd
        # y = x @ W_dec + b_dec
        return x @ self.W_dec + self.b_dec

# %% Sandwich Function Definition

def sandwich(
    in_feat_mags : Float[Tensor, "... feat_in"],
    sae_mlpin : DummySAE,
    mlp : MLP,
    sae_mlpout : DummySAE,
    normalize : Callable[[Tensor], Tensor]
) -> Float[Tensor, "... feat_out"]:
    """
    Computes the 'sandwich' operation:
    in_feats -> normalize -> decode -> MLP -> skip_add -> encode -> out_feats
    """
    # 1. Normalize input features
    normalized_input = normalize(in_feat_mags)

    # 2. Decode normalized features into residual stream embedding space
    resid_mid = sae_mlpin.decode(normalized_input)

    # 3. Pass through MLP
    mlp_post = mlp(resid_mid)

    # 4. Add skip connection from pre-MLP residual stream
    resid_post = mlp_post + resid_mid

    # 5. Encode the result back into feature space
    resid_post_featmag = sae_mlpout.encode(resid_post)

    return resid_post_featmag

# %% Hook Function for MLP Activation Gradients

# Global dictionary to store activations/gradients from hooks
activations_store: Dict[str, Tensor] = {}

def mlp_gelu_hook_fn(module, inputs, outputs):
    """Hook to capture the gradient of the GELU activation."""
    pre_actfn = inputs[0]
    # post_actfn = outputs # Not needed directly

    # Calculate derivative g'(pre_actfn)
    pre_actfn_copy = pre_actfn.detach().requires_grad_(True)
    with torch.enable_grad():
        # Recompute GELU to allow autograd to track it
        recomputed_post_actfn = F.gelu(pre_actfn_copy, approximate="tanh")
        # Calculate gradient of the output w.r.t input for each element
        # We want d(gelu(z))/dz evaluated at z=pre_actfn
        grad_of_actfn = torch.autograd.grad(
            outputs=recomputed_post_actfn,
            inputs=pre_actfn_copy,
            grad_outputs=torch.ones_like(recomputed_post_actfn), # d(sum(y)) / dx_i = dy_i / dx_i for elementwise
            retain_graph=False,
            create_graph=False # No need for higher order derivatives
        )[0]

    activations_store["mlp_gelu_grad"] = grad_of_actfn # Store g'
    # The hook should return the original outputs unmodified
    return outputs

# %% Helper Function for Elementwise Derivatives

def get_elementwise_derivative(
    func: Callable[[Tensor], Tensor],
    x: Float[Tensor, "... d"]
) -> Float[Tensor, "... d"]:
    """Computes the elementwise derivative of func w.r.t. x using autograd."""
    x_detached = x.detach().requires_grad_(True)
    with torch.enable_grad():
        y = func(x_detached)
        # Calculate dy/dx elementwise
        grads = torch.autograd.grad(
            outputs=y, # Gradient of each output element
            inputs=x_detached,
            grad_outputs=torch.ones_like(y), # Propagate 1 for each output element
            create_graph=False,
            retain_graph=False,
        )[0]
    return grads

# %% Analytical Jacobian Calculation Function

def all_jacobian_normalized_input(
    sae_mlpin : DummySAE,
    sae_mlpout : DummySAE,
    mlp: MLP,
    mlp_act_grads: Float[Tensor, "batch seq d_mlp"], # Derivative g' from hook
    normalize_func: Callable[[Tensor], Tensor],
    in_feat_mags: Float[Tensor, "batch seq feat_in"], # Original input needed for norm'
) -> Float[Tensor, "batch seq feat_out feat_in"]:
    """
    Computes the analytical Jacobian d(output_features)/d(input_features)
    for the sandwich function with input normalization.
    """
    # --- 1. Calculate Derivative of the Normalization Function (norm') ---
    norm_grads = get_elementwise_derivative(normalize_func, in_feat_mags)
    # Shape: (batch, seq, feat_in)

    # --- 2. Define Model Weights (matching mathematical convention) ---
    # SAEs: W_dec (feat_in, n_embd), W_enc (n_embd, feat_out)
    W_dec_in = sae_mlpin.W_dec
    W_enc_out = sae_mlpout.W_enc
    # MLP: W_in (d_mlp, n_embd), W_out (n_embd, d_mlp) for nn.Linear
    # Need W_in.T (n_embd, d_mlp) and W_out.T (d_mlp, n_embd) for forward pass calc
    # Need W_in (d_mlp, n_embd) and W_out (n_embd, d_mlp) for derivative chain rule
    W_in = mlp.W_in.weight  # Shape (4*n_embd, n_embd)
    W_out = mlp.W_out.weight # Shape (n_embd, 4*n_embd)

    # --- 3. Calculate Jacobian before Normalization (J_prev = d(output)/d(norm_input)) ---
    # J_prev = d(out)/d(res_post) @ d(res_post)/d(res_mid) @ d(res_mid)/d(norm_in)
    # d(out)/d(res_post) = W_enc_out.T  (feat_out, n_embd)
    # d(res_mid)/d(norm_in) = W_dec_in.T (n_embd, feat_in)
    # d(res_post)/d(res_mid) = d(mlp_post)/d(res_mid) + I
    # d(mlp_post)/d(res_mid) = d(mlp_post)/d(mlp_act) @ d(mlp_act)/d(mlp_pre) @ d(mlp_pre)/d(res_mid)
    #                      = W_out @ diag(g') @ W_in
    # So, J_prev = W_enc_out.T @ (W_out @ diag(g') @ W_in + I) @ W_dec_in.T
    # J_prev = (W_enc_out.T @ W_out @ diag(g') @ W_in @ W_dec_in.T) + (W_enc_out.T @ W_dec_in.T)

    # MLP Path Component: W_enc_out.T @ W_out @ diag(g') @ W_in @ W_dec_in.T
    TermA = W_enc_out.T @ W_out              # (feat_out, n_embd) @ (n_embd, d_mlp) -> (feat_out, d_mlp)
    TermB = W_in @ W_dec_in.T                # (d_mlp, n_embd) @ (n_embd, feat_in) -> (d_mlp, feat_in)

    dtype = TermA.dtype
    jacobian_mlp_part = einops.einsum(
        TermA,                      # (feat_out, d_mlp)
        mlp_act_grads.to(dtype),    # (batch, seq, d_mlp) - Diagonal g'
        TermB,                      # (d_mlp, feat_in)
        "fo d, b s d, d fi -> b s fo fi", # sum over d_mlp index 'd'
    )
    # Shape: (batch, seq, feat_out, feat_in)

    # Skip Path Component: W_enc_out.T @ W_dec_in.T
    jacobian_skip_part = W_enc_out.T @ W_dec_in.T
    # Shape: (feat_out, feat_in)

    # Combine MLP and Skip paths (Jacobian before input norm derivative)
    jacobian_prev = jacobian_mlp_part + jacobian_skip_part.to(jacobian_mlp_part.dtype) # Broadcast skip part
    # Shape: (batch, seq, feat_out, feat_in)

    # --- 4. Apply Normalization Derivative via Chain Rule ---
    # J = J_prev @ diag(norm'(x))
    # Scale column fi of J_prev by norm_grads[b, s, fi]
    total_jacobian = jacobian_prev * norm_grads.unsqueeze(-2) # (b,s,fo,fi) * (b,s, 1,fi)
    # Shape: (batch, seq, feat_out, feat_in)

    return total_jacobian


# %% Main Execution Block

if __name__ == "__main__":
    # --- Hyperparameters ---
    n_embd = 8      # Embedding dimension
    n_feat = 1024     # Number of SAE features (for both input and output SAE)
    batch_size = 2
    seq_len = 4

    # --- Instantiate Models ---
    gpt_config = GPTConfig(n_embd=n_embd)
    mlp = MLP(gpt_config).to(device)
    # Use same feature size for input and output SAEs for simplicity
    sae_mlpin = DummySAE(n_embd, n_feat).to(device)
    sae_mlpout = DummySAE(n_embd, n_feat).to(device)

    # --- Input Data ---
    # Generate random input feature magnitudes
    in_feat_mags = torch.randn(batch_size, seq_len, n_feat, device=device)

    # --- Define Normalization Function ---
    def normalize_fn(x):
        return F.gelu(x, approximate="tanh") # Use GELU for this example

    # --- Run Forward Pass with Hook ---
    # Clear any previous hook data
    activations_store.clear()
    # Register the hook on the MLP's GELU module
    hook_handle = mlp.gelu.register_forward_hook(mlp_gelu_hook_fn)

    # Perform the forward pass (this triggers the hook)
    # We don't need the output here, just need the hook to run
    _ = sandwich(in_feat_mags, sae_mlpin, mlp, sae_mlpout, normalize_fn)

    # Remove the hook now that we have the gradients
    hook_handle.remove()

    # Check if hook ran successfully
    if "mlp_gelu_grad" not in activations_store:
        raise RuntimeError("Hook did not capture MLP GELU gradients.")
    mlp_gelu_grads = activations_store["mlp_gelu_grad"]
    print(f"Captured mlp_gelu_grads shape: {mlp_gelu_grads.shape}") # Should be (batch, seq, 4*n_embd)

    # --- Calculate Analytical Jacobian ---
    print("\nCalculating Analytical Jacobian...")
    jacobian_analytical = all_jacobian_normalized_input(
        sae_mlpin,
        sae_mlpout,
        mlp,
        mlp_gelu_grads,
        normalize_fn,
        in_feat_mags
    )
    print(f"Analytical Jacobian shape: {jacobian_analytical.shape}")

    # --- Calculate Jacobian using Autograd Functional ---
    print("\nCalculating Autograd Jacobian...")

    # Define the function for torch.autograd.functional.jacobian
    # It needs to take only the input tensor we're differentiating w.r.t.
    def compute_output_fixed_models(input_features):
        # Models and normalize_fn are accessed from the outer scope
        return sandwich(
            input_features,
            sae_mlpin,
            mlp,
            sae_mlpout,
            normalize_fn
        )

    # Make sure input requires grad for the autograd calculation
    in_feat_mags_grad = in_feat_mags.clone().detach().requires_grad_(True)

    # Compute the Jacobian
    # jacobian output shape is (out_dims..., in_dims...)
    # Output shape: (batch, seq, feat_out)
    # Input shape: (batch, seq, feat_in)
    # Expected Jacobian shape: (batch, seq, feat_out, batch, seq, feat_in)
    # We need vectorization=True for batch dims
    jacobian_autograd = torch.autograd.functional.jacobian(
        func=compute_output_fixed_models,
        inputs=in_feat_mags_grad,
        create_graph=False,
        vectorize=True # Important for batch processing!
    )
    print(f"Raw Autograd Jacobian shape: {jacobian_autograd.shape}")

    # Rearrange autograd Jacobian to match analytical: (batch, seq, feat_out, feat_in)
    # The output dims are (batch, seq, feat_out)
    # The input dims are (batch, seq, feat_in)
    # vectorize=True gives (batch, seq, feat_out, feat_in) directly IF the function maps (B, S, Fin) -> (B, S, Fout) elementwise across B, S. Let's check.
    # Yes, our function processes batch and seq independently.
    # If vectorize=True wasn't used, or didn't work as expected, we might get (B, S, Fout, B, S, Fin)
    # and would need something like:
    # jacobian_autograd = torch.diagonal(jacobian_autograd, dim1=0, dim2=3).permute(2, 0, 1, 3) # B, S, Fout, Fin
    # jacobian_autograd = torch.diagonal(jacobian_autograd, dim1=1, dim2=3) # B, Fout, S, Fin <- need careful checking

    # Let's assume vectorize=True gives the desired (B, S, Fout, Fin)
    # If shapes don't match, the assert_close will fail, indicating a problem here.
    if jacobian_autograd.shape != jacobian_analytical.shape:
         print("WARNING: Autograd Jacobian shape doesn't match analytical after vectorize=True.")
         # Attempt manual diagonal extraction if shapes mismatch (common issue)
         # We want J[b, s, fo, fi] = d(out[b,s,fo]) / d(in[b,s,fi])
         # Raw shape might be (B,S,Fout, B,S,Fin)
         try:
            print("Attempting manual diagonal extraction...")
            jacobian_autograd_fixed = torch.zeros_like(jacobian_analytical)
            for b in range(batch_size):
                for s in range(seq_len):
                    # Extract the block d(out[b,s,:]) / d(in[b,s,:])
                    jacobian_autograd_fixed[b, s] = jacobian_autograd[b, s, :, b, s, :]
            jacobian_autograd = jacobian_autograd_fixed
            print(f"Reshaped Autograd Jacobian shape: {jacobian_autograd.shape}")
         except Exception as e:
            print(f"Manual diagonal extraction failed: {e}")
            print("Proceeding with potentially incorrect shape for comparison.")


    # --- Compare Results ---
    print("\nComparing Jacobians...")
    torch.testing.assert_close(jacobian_autograd, jacobian_analytical, rtol=1e-4, atol=1e-5)
    print("✅ Jacobians match!")
    
# %% Run the script
# (This line is typically executed in an interactive environment like Jupyter)
# If running as a .py file, the `if __name__ == "__main__":` block handles execution.