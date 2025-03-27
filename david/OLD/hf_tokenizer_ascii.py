# %%
import os
import sys
from transformer_lens import HookedTransformer, HookedTransformerConfig
# Add root directory to sys.path dynamically
# sys.path hack to run in VSCode interactive session
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..",)))
print("\n".join(sys.path))

from config.gpt.training import options
from models.gpt import GPT
from data.tokenizers import ASCIITokenizer, TikTokenTokenizer
from transformers import PreTrainedTokenizerBase

# %%
ascii_tokenizer = ASCIITokenizer()
# %%
class ASCIITokenizer(PreTrainedTokenizerBase):
    def __init__(self):
        super().__init__(max_len = 128,
                         padding_side = "right",
                         pad_token = "?",
                         eos_token = "?",
                         bos_token = "?",
                         unk_token = "?",
                        
                         )
        self.vocab = {chr(i): i for i in range(128)}
        self.id_to_token = {i: token for token, i in self.vocab.items()}
        self.token_to_id = self.vocab

# Instantiate your custom tokenizer

# %%
