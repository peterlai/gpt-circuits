from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from circuits import Circuit
from circuits.search.ablation import ResampleAblator
from data.tokenizers import Tokenizer
from models.sparsified import SparsifiedGPT


@dataclass(frozen=True)
class Divergence:
    kl_divergence: float
    predictions: dict[str, float]

def analyze_token_mask_divergence(
    model: SparsifiedGPT,
    ablator: ResampleAblator,
    layer_idx: int,
    target_token_idx: int,
    target_logits: torch.Tensor,  # Shape: (V)
    token_mask: torch.Tensor,  # Shape: (T)
    feature_magnitudes: torch.Tensor,  # Shape: (T, F)
    num_samples: int,
) -> Divergence:
    """
    Calculate KL divergence between target logits and logits produced through use of a token mask.
    """
    # Create a feature mask from the token mask
    feature_mask = torch.zeros_like(feature_magnitudes, dtype=torch.bool)
    feature_mask[token_mask] = True

    # Patch feature magnitudes
    patched_feature_magnitudes = ablator.patch(
        layer_idx=layer_idx,
        target_token_idx=target_token_idx,
        feature_magnitudes=feature_magnitudes.cpu().numpy(),
        feature_mask=feature_mask.cpu().numpy(),
        num_samples=num_samples,
    )

    # Get predicted logits when using patched feature magnitudes
    predicted_logits = get_predicted_logits(
        model,
        layer_idx,
        torch.tensor(patched_feature_magnitudes, device=feature_magnitudes.device),
        target_token_idx,
    )

    # Compute KL divergence
    kl_div = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(predicted_logits, dim=-1),
        torch.nn.functional.softmax(target_logits, dim=-1),
        reduction="sum",
    )

    # Calculate predictions
    predictions = get_predictions(model.gpt.config.tokenizer, predicted_logits)

    # Return results
    return Divergence(kl_divergence=kl_div.item(), predictions=predictions)


def analyze_circuit_divergence(
    model: SparsifiedGPT,
    ablator: ResampleAblator,
    layer_idx: int,
    target_token_idx: int,
    target_logits: torch.Tensor,  # Shape: (V)
    circuit_variants: Sequence[Circuit],  # List of circuit variants
    feature_magnitudes: torch.Tensor,  # Shape: (T, F)
    num_samples: int,
) -> dict[Circuit, Divergence]:
    """
    Calculate KL divergence between target logits and logits produced through use of circuit nodes on a single layer.

    :param model: The sparsified model to use for circuit extraction.
    :param ablator: Ablation tecnique to use for circuit extraction.
    :param layer_idx: The layer index to use for circuit extraction.
    :param target_token_idx: The token index to use for circuit extraction.
    :param target_logits: The target logits for the target token.
    :param circuit_variants: The circuit variants to use for circuit extraction.
    :param feature_magnitudes: Feature magnitudes to use for each circuit variant.
    :param num_samples: The number of samples to use for ablation.
    """
    # For storing results
    results: dict[Circuit, Divergence] = {}

    # Patch feature magnitudes for each circuit variant
    patched_feature_magnitudes = patch_feature_magnitudes(
        ablator,
        layer_idx,
        target_token_idx,
        circuit_variants,
        feature_magnitudes.cpu().numpy(),
        num_samples=num_samples,
    )

    # Get predicted logits for each circuit variant when using patched feature magnitudes
    predicted_logits = {}
    for circuit_variant, magnitudes in patched_feature_magnitudes.items():
        predicted_logits[circuit_variant] = get_predicted_logits(
            model,
            layer_idx,
            torch.tensor(magnitudes, device=feature_magnitudes.device),
            target_token_idx,
        )

    # Calculate KL divergence and predictions for each variant
    for circuit_variant, circuit_logits in predicted_logits.items():
        # Compute KL divergence
        kl_div = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(circuit_logits, dim=-1),
            torch.nn.functional.softmax(target_logits, dim=-1),
            reduction="sum",
        )

        # Calculate predictions
        predictions = get_predictions(model.gpt.config.tokenizer, circuit_logits)

        # Store results
        results[circuit_variant] = Divergence(kl_divergence=kl_div.item(), predictions=predictions)

    return results


def patch_feature_magnitudes(
    ablator: ResampleAblator,
    layer_idx: int,
    target_token_idx: int,
    circuit_variants: Sequence[Circuit],
    feature_magnitudes: np.ndarray,  # Shape: (T, F)
    num_samples: int,
) -> dict[Circuit, np.ndarray]:  # Shape: (num_samples, T, F)
    """
    Patch feature magnitudes for a list of circuit variants.
    """
    # For mapping variants to patched feature magnitudes
    patched_feature_magnitudes: dict[Circuit, np.ndarray] = {}

    # Patch feature magnitudes for each variant
    with ThreadPoolExecutor() as executor:
        futures: dict[Future, Circuit] = {}
        for circuit_variant in circuit_variants:
            # Create feature mask
            feature_mask = np.zeros_like(feature_magnitudes, dtype=bool)
            layer_nodes = [n for n in circuit_variant.nodes if n.layer_idx == layer_idx]
            if layer_nodes:
                token_indices = torch.tensor([node.token_idx for node in layer_nodes])
                feature_indices = torch.tensor([node.feature_idx for node in layer_nodes])
                feature_mask[token_indices, feature_indices] = True
            # Patch feature magnitudes
            future = executor.submit(
                ablator.patch,
                layer_idx=layer_idx,
                target_token_idx=target_token_idx,
                feature_magnitudes=feature_magnitudes,
                feature_mask=feature_mask,
                num_samples=num_samples,
            )
            futures[future] = circuit_variant

        for future in as_completed(futures):
            circuit_variant = futures[future]
            patched_feature_magnitudes[circuit_variant] = future.result()

    # Return patched feature magnitudes
    return patched_feature_magnitudes


@torch.no_grad()
def get_predicted_logits(
    model: SparsifiedGPT,
    layer_idx: int,
    feature_magnitudes: torch.Tensor,  # Shape: (num_samples, T, F)
    target_token_idx: int,
) -> torch.Tensor:  # Shape: (V)
    """
    Get predicted logits when using patched feature magnitudes.

    TODO: Use batching to improve performance
    """
    # Reconstruct activations
    x_reconstructed = model.saes[str(layer_idx)].decode(feature_magnitudes)  # type: ignore

    # Compute logits
    predicted_logits = model.gpt.forward_with_patched_activations(
        x_reconstructed, layer_idx=layer_idx
    )  # Shape: (num_samples, T, V)

    # We only care about logits for the target token
    predicted_logits = predicted_logits[:, target_token_idx, :]  # Shape: (num_samples, V)

    # Convert logits to probabilities before averaging across samples
    predicted_probabilities = torch.nn.functional.softmax(predicted_logits, dim=-1)
    predicted_probabilities = predicted_probabilities.mean(dim=0)  # Shape: (V)
    predicted_logits = torch.log(predicted_probabilities)  # Shape: (V)

    return predicted_logits


@torch.no_grad()
def compute_downstream_magnitudes(
    model: SparsifiedGPT,
    layer_idx: int,
    patched_feature_magnitudes: torch.Tensor,  # Shape: (num_samples, T, F)
) -> dict[Circuit, torch.Tensor]:  # Shape: (num_sample, T, F)
    """
    Get downstream feature magnitudes for a set of circuit variants when using patched feature magnitudes.

    TODO: Use batching to improve performance
    """
    # Reconstruct activations
    x_reconstructed = model.saes[str(layer_idx)].decode(patched_feature_magnitudes)  # type: ignore

    # Compute downstream activations
    x_downstream = model.gpt.transformer.h[layer_idx](x_reconstructed)  # type: ignore

    # Encode to get feature magnitudes
    downstream_sae = model.saes[str(layer_idx + 1)]
    downstream_feature_magnitudes = downstream_sae(x_downstream).feature_magnitudes  # Shape: (num_sample, T, F)
    return downstream_feature_magnitudes


def get_predictions(
    tokenizer: Tokenizer,
    logits: torch.Tensor,  # Shape: (V)
    count: int = 5,
) -> dict[str, float]:
    """
    Map logits to probabilities and return top 5 predictions.
    """
    probs = torch.nn.functional.softmax(logits, dim=-1)
    topk = torch.topk(probs, k=count)
    results: dict[str, float] = {}
    for i, p in zip(topk.indices, topk.values):
        results[tokenizer.decode_token(int(i.item()))] = round(p.item() * 100, 2)
    return results