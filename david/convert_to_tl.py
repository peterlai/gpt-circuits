# %%


import os
import sys
from transformer_lens import HookedTransformer, HookedTransformerConfig
# Add root directory to sys.path dynamically
# sys.path hack to run in VSCode interactive session
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..",)))
print("\n".join(sys.path))
# %%
from config.gpt.training import options
from models.gpt import GPT
from data.tokenizers import ASCIITokenizer, TikTokenTokenizer
import torch
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer

from david.utils import generate

def define_model(name):
    config = options[name]

    # Load the GPT model
    model = GPT(config.gpt_config)
    model_path = os.path.join("../checkpoints", name)
    model = model.load(model_path, device=config.device)
    model.to(config.device)

    # Initialize the tokenizer
    tokenizer = ASCIITokenizer() if "shake" in name else TikTokenTokenizer()
    tokenizer.eos_token_id = ord("?")
    tokenizer.pad_token_id = ord("?")

    return model, tokenizer, config

def convert_gpt_to_transformer_lens(model, config):
    # Create a HookedTransformerConfig
    hooked_config = HookedTransformerConfig(
        n_layers=config.gpt_config.n_layer,
        d_model=config.gpt_config.n_embd,
        n_ctx=config.gpt_config.block_size,
        d_head=config.gpt_config.n_embd // config.gpt_config.n_head,
        d_mlp=4 * config.gpt_config.n_embd,
        n_heads=config.gpt_config.n_head,
        model_name=config.gpt_config.name,
        device=config.device,
        act_fn="gelu_pytorch_tanh",
        d_vocab=config.gpt_config.vocab_size,
        use_attn_result=True,
        use_local_attn=False,
        tokenizer_prepends_bos=True,
        default_prepend_bos=True,
    )
    # Initialize the HookedTransformer
    ht = HookedTransformer(hooked_config)

    class ASCIITokenizer(PreTrainedTokenizer):
        """
        Tokenizer that treats each character in the input as a token, conforming to Hugging Face's PreTrainedTokenizer.
        """
        def __init__(self, tokenizer_prepend_bos=False, **kwargs):
            super().__init__(**kwargs)
            self.add_special_tokens({})
            self.pad_token = "?"
            self.eos_token = "?"
        
        @property
        def vocab_size(self):
            return 128

        def _tokenize(self, text):
            return [c if ord(c) < 128 else "?" for c in text]
        
        def _convert_token_to_id(self, token):
            return ord(token)
        
        def _convert_id_to_token(self, index):
            return chr(index)
        
        def convert_tokens_to_string(self, tokens):
            return "".join(tokens)
        
        def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
            return token_ids_0 if token_ids_1 is None else token_ids_0 + token_ids_1
        
        def get_vocab(self):
            return {chr(i): i for i in range(128)}
        
        def decode_sequence(self, tokens):
            return self.decode(tokens)
        
        def __call__(self, text, **kwargs):
            return {
                'input_ids': self.encode(text, **kwargs),
                'attention_mask': torch.ones_like(self.encode(text, **kwargs))
            }

    custom_tokenizer = ASCIITokenizer()
    ht.tokenizer = custom_tokenizer

    from transformer_lens.pretrained.weight_conversions import convert_nanogpt_weights

    # Convert weights using the provided function
    state_dict = convert_nanogpt_weights(model.state_dict(), hooked_config)
    for l in range(hooked_config.n_layers):
        state_dict[f'blocks.{l}.attn.IGNORE'] = ht.blocks[l].attn.IGNORE
        state_dict[f'blocks.{l}.attn.mask'] = ht.blocks[l].attn.mask
        state_dict[f'unembed.b_U'] = torch.zeros_like(ht.unembed.b_U)
    # Load the converted weights into the HookedTransformer
    ht.load_state_dict(state_dict)

    return ht

def run_tests(model, ht, config):
    # Generate tokens and resid once
    tokens = torch.randint(0, 100, (1, 10)).to(config.device)
    resid = torch.randn((13,7,64)).to(config.device)

    def check_all_close_and_print(expected, actual, component_name):
        if not torch.allclose(expected, actual, atol=1e-6, rtol=1e-6):
            abs_diff = torch.abs(expected - actual).max().item()
            rel_diff = (abs_diff / torch.abs(expected).max()).item()
            print(f"{component_name} test failed: abs_diff={abs_diff}, rel_diff={rel_diff}")
        else:
            print(f"{component_name} test passed")

    # Perform tests directly using check_all_close_and_print
    # Test embedding
    check_all_close_and_print(
        model.transformer.wte(tokens),
        ht.embed(tokens),
        "embed"
    )

    # Test positional embedding
    _, T = tokens.shape
    pos = torch.arange(0, T, dtype=torch.long, device=tokens.device)
    check_all_close_and_print(
        model.transformer.wpe(pos),
        ht.pos_embed(tokens),
        "pos_embed"
    )

    # Test layer normalization
    check_all_close_and_print(
        model.transformer.ln_f(resid),
        ht.ln_final(resid),
        "ln_final"
    )

    # Test unembedding
    check_all_close_and_print(
        model.lm_head(resid),
        ht.unembed(resid),
        "unembed"
    )

    # Test each layer
    for l in range(model.config.n_layer):
        print(f"layer {l}")
        check_all_close_and_print(
            model.transformer.h[l].ln_1(resid),
            ht.blocks[l].ln1(resid),
            f"ln1 layer {l}"
        )
        check_all_close_and_print(
            model.transformer.h[l].ln_2(resid),
            ht.blocks[l].ln2(resid),
            f"ln2 layer {l}"
        )
        check_all_close_and_print(
            model.transformer.h[l].mlp(resid),
            ht.blocks[l].mlp(resid),
            f"mlp layer {l}"
        )
        check_all_close_and_print(
            model.transformer.h[l].attn(resid),
            ht.blocks[l].attn(resid, resid, resid),
            f"attn layer {l}"
        )

    # Test full transformer
    check_all_close_and_print(
        model(tokens)[0],
        ht(tokens),
        "full transformer"
    )

    print("="*80)
    print(generate(model, ht.tokenizer, "Hello, how are you?", max_length=100, temperature=0.00001))
    print("="*80)
    print(generate(ht, ht.tokenizer, "Hello, how are you?", max_length=100, temperature=0.00001))
    print("="*80)

# %%
if __name__ == "__main__":
    name = 'shakespeare_64x4'
    model, tokenizer, config = define_model(name)
    ht = convert_gpt_to_transformer_lens(model, config)
    run_tests(model, ht, config)
# %%