from dataclasses import dataclass, field

from config import TrainingConfig, map_options

from .models import GPTConfig, gpt_options


@dataclass
class GPTTrainingConfig(TrainingConfig):
    gpt_config: GPTConfig = field(default_factory=GPTConfig)


# Training configuration options
options: dict[str, GPTTrainingConfig] = map_options(
    GPTTrainingConfig(
        name="shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        data_dir="data/shakespeare",
        eval_interval=250,
        eval_steps=100,
        batch_size=128,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        warmup_steps=1000,
        max_steps=10000,
        decay_lr=True,
        min_lr=1e-5,
    ),
    GPTTrainingConfig(
        name="shakespeare_64x4_dyt",
        gpt_config=gpt_options["ascii_64x4_dyt"],
        data_dir="data/shakespeare",
        eval_interval=250,
        eval_steps=100,
        batch_size=128,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        warmup_steps=1000,
        max_steps=20_000,
        decay_lr=True,
        min_lr=1e-5,
    ),
    GPTTrainingConfig(
        name="shakespeare_128x6",
        gpt_config=gpt_options["ascii_128x6"],
        data_dir="data/shakespeare",
        eval_interval=250,
        eval_steps=100,
        batch_size=128,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        warmup_steps=300,
        max_steps=3000,
        decay_lr=True,
        min_lr=1e-4,
    ),
    GPTTrainingConfig(
        name="stories_32x4",
        gpt_config=gpt_options["tiktoken_32x4"],
        data_dir="data/tiny_stories_10m",
        eval_interval=250,
        eval_steps=100,
        batch_size=128,
        gradient_accumulation_steps=8,
        learning_rate=1e-3,
        max_steps=5000,
    ),
    GPTTrainingConfig(
        name="stories_256x4",
        gpt_config=gpt_options["tiktoken_256x4"],
        data_dir="data/tiny_stories",
        eval_interval=250,
        eval_steps=100,
        batch_size=512,
        gradient_accumulation_steps=8,
        learning_rate=1e-3,
        max_steps=5000,
    ),
)
