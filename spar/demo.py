# %%


import os
import sys

# Add root directory to sys.path dynamically
# sys.path hack to run in VSCode interactive session
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..",)))
print("\n".join(sys.path))
# %%
from config.gpt.training import options
from models.gpt import GPT
from data.tokenizers import ASCIITokenizer, TikTokenTokenizer

from utils import generate
name = 'shakespeare_64x4'
config = options[name]

model = GPT(config.gpt_config)
model_path = os.path.join("../checkpoints", name)
model = model.load(model_path, device=config.device)

tokenizer = ASCIITokenizer() if "shake" in name else TikTokenTokenizer()

print(generate(model, tokenizer, "Today I thought,", max_length=100))


# %%
