import torch

class MaxSizeList():
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.data = []

    def append(self, item):
        #item is a tuple of index and value
        #add item to list using bisect to preserve sorted order
        #if size is too large remove the first element
        bisect.insort(self.data, item, key=lambda x: x[1])
        if len(self.data) > self.max_size:
            self.data.pop(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    def _get_valueless_list(self):
        #return a list of the indices in the data
        return [x[0] for x in self.data]

    def __iter__(self):
        return iter(self.data)


def sorted_indices_by_value(tensor: torch.Tensor) -> list[tuple[int, int]]:
    assert tensor.ndim == 2, "Input must be a 2D tensor"

    # Flatten and get sorted indices
    flat_indices = torch.argsort(tensor.view(-1))

    # Convert flat indices back to 2D (i, j)
    indices = [divmod(idx.item(), tensor.size(1)) for idx in flat_indices]

    return indices

def get_SAE_activations(model, data, layers):
    if model.config.sae_variant.startswith('jsae') or model.config.sae_variant.startswith('mlp'):
        return get_SAE_activations_MLP(model, data, layers)
    else:  
        return get_SAE_activations_BLOCK(model, data, layers)


@torch.no_grad()
def get_SAE_activations_BLOCK(model, data, layers):
    # Get the activations for the specified layers
    activations = {}
    max_layer = max(layers)
    assert max_layer <= model.gpt.config.n_layer, f"Layer {max_layer} is out of range for the model with {model.gpt.config.n_layer} layers"
    B, T = data.size()
    assert (
        T <= model.config.block_size
    ), f"Cannot forward sequence of length {T}, block size is only {model.config.block_size}"
    # forward the token and posisition embeddings
    pos = torch.arange(0, T, dtype=torch.long, device=data.device)  # shape (T)
    pos_emb = model.gpt.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
    tok_emb = model.gpt.transformer.wte(data)  # token embeddings of shape (B, T, n_embd)

    x = tok_emb + pos_emb
    layer = 0

    while layer <= max_layer:
        if layer in layers:
            activations[layer] = model.saes[f'{layer}'].encode(x)
        if layer < model.gpt.config.n_layer:
            x = model.gpt.transformer.h[layer](x)
            layer += 1
        else:
            break
    return activations

@torch.no_grad()
def get_SAE_activations_MLP(model, data, layers):
    # Get the activations for the specified layers
    activations = {}
    max_layer = max(layers)
    assert max_layer < 2*model.gpt.config.n_layer, f"SAE Layer {max_layer} is out of range for the model with {2*model.gpt.config.n_layer} SAE layers "
    B, T = data.size()
    assert (
        T <= model.config.block_size
    ), f"Cannot forward sequence of length {T}, block size is only {model.config.block_size}"
    # forward the token and posisition embeddings
    pos = torch.arange(0, T, dtype=torch.long, device=data.device)  # shape (T)
    pos_emb = model.gpt.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
    tok_emb = model.gpt.transformer.wte(data)  # token embeddings of shape (B, T, n_embd)

    x = tok_emb + pos_emb
    model_layer = 0
    while model_layer < model.gpt.config.n_layer and 2*model_layer +1 <= max_layer:
        block = model.gpt.transformer.h[model_layer]

        x = x + block.attn(block.ln_1(x))
        
        y = block.ln_2(x)

        if 2*model_layer in layers:
            activations[2*model_layer] = model.saes[f'{2*model_layer}'].encode(y)
        y = block.mlp(y)
        if 2*model_layer + 1 in layers:
            activations[2*model_layer + 1] = model.saes[f'{2*model_layer + 1}'].encode(y)
        x = x + y
        model_layer += 1
    return activations


    

    