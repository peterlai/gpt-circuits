import torch

def generate(model, tokenizer, prompt, max_length=50, temperature=0.7) -> str:
    """
    Generate text from a prompt using the model
    """
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt)
    tokens = torch.Tensor(tokens).long().unsqueeze(0).to(device)
    for _ in range(max_length):
        output = model(tokens)
        if isinstance(output, tuple):
            output = output[0]
        logits = output[:, -1]
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=-1)
    return tokenizer.decode_sequence(tokens[0].tolist())