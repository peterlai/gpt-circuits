# %%

import torch
import os
import sys
import os
from data.tokenizers import ASCIITokenizer, TikTokenTokenizer

# Add root directory to sys.path dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..",)))
#print("\n".join(sys.path))


from config.gpt.training import options
from models.gpt import GPT
checkpoints = "../checkpoints"
name = 'tiny_64x2'
#name = 'shakespeare_64x4'
config = options[name]

model = GPT(config.gpt_config)
model_path = os.path.join(checkpoints, name)
model = model.load(model_path, device=config.device)

tokenizer = ASCIITokenizer() if "shake" in name else TikTokenTokenizer()

def generate(model, tokenizer, prompt, max_length=10, temperature=0.7) -> str:
    """
    Generate text from a prompt using the model
    """
    tokens = tokenizer.encode(prompt)
    tokens = torch.Tensor(tokens).long().unsqueeze(0)
    for _ in range(max_length):
        logits = model(tokens)[0][:, -1, :]
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=-1)
    return tokenizer.decode_sequence(tokens[0].tolist())

print(generate(model, tokenizer, "Today I thought,", max_length=100))


# %%
