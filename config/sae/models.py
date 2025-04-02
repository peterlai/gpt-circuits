from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from config import Config, map_options
from config.gpt.models import GPTConfig, gpt_options


class SAEVariant(str, Enum):
    STANDARD = "standard"
    STANDARD_V2 = "standard_v2"
    GATED = "gated"
    GATED_V2 = "gated_v2"
    JUMP_RELU = "jumprelu"
    JUMP_RELU_STAIRCASE = "jumprelu_staircase"
    TOPK = "topk"
    TOPK_STAIRCASE = "topk_staircase"
    TOPK_STAIRCASE_DETACH = "topk_staircase_detach"


@dataclass
class SAEConfig(Config):
    gpt_config: GPTConfig = field(default_factory=GPTConfig)
    n_features: tuple = ()  # Number of features in each layer
    sae_variant: SAEVariant = SAEVariant.STANDARD
    top_k: Optional[tuple[int, ...]] = None  # Required for topk variant

    @property
    def block_size(self) -> int:
        return self.gpt_config.block_size

    @staticmethod
    def dict_factory(fields: list) -> dict:
        """
        Only export n_features and sae_variant.
        """
        whitelisted_fields = ("n_features", "sae_variant", "top_k")
        return {k: v for (k, v) in fields if k in whitelisted_fields and v is not None}


# SAE configuration options
sae_options: dict[str, SAEConfig] = map_options(
    SAEConfig(
        name="standardx16.tiny_32x4",
        gpt_config=gpt_options["tiktoken_32x4"],
        n_features=tuple(32 * n for n in (16, 16, 16, 16, 16)),
        sae_variant=SAEVariant.STANDARD,
    ),
    # No such thing as tiny_64x2?
    # SAEConfig(
    #     name="standardx8.tiny_64x2",
    #     gpt_config=gpt_options["tiktoken_64x2"],
    #     n_features=tuple(64 * n for n in (8, 8, 8, 8, 8)),
    #     sae_variant=SAEVariant.STANDARD,
    # ),
    SAEConfig(
        name="standardx8.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 8, 8, 8, 8)),
        sae_variant=SAEVariant.STANDARD,
    ),
    SAEConfig(
        name="topk-10-x8.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 8, 8, 8, 8)),
        sae_variant=SAEVariant.TOPK,
        top_k=(10, 10, 10, 10, 10),
    ),
    SAEConfig(
        name="topk-staircase-10-x8-noshare.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 16, 24, 32, 40)),
        sae_variant=SAEVariant.TOPK,
        top_k=(10, 10, 10, 10, 10),
    ),
    SAEConfig(
        name="topk-staircase-10-x8-share.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 16, 24, 32, 40)),
        sae_variant=SAEVariant.TOPK_STAIRCASE,
        top_k=(10, 10, 10, 10, 10),
    ),
    SAEConfig(
        name="topk-staircase-10-x8-detach.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 16, 24, 32, 40)),
        sae_variant=SAEVariant.TOPK_STAIRCASE_DETACH,
        top_k=(10, 10, 10, 10, 10),
    ),
    SAEConfig(
        name="standardx40.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (40, 40, 40, 40, 40)),
        sae_variant=SAEVariant.STANDARD,
    ),
    SAEConfig(
        name="staircasex8.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 16, 24, 32, 40)),
        sae_variant=SAEVariant.STANDARD,
    ),
    SAEConfig(
        name="topk-x8.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 8, 8, 8, 8)),
        sae_variant=SAEVariant.TOPK,
        top_k=(10,10,10,10,10),
    ),
    SAEConfig(
        name="top5.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 8, 8, 8, 8)),
        sae_variant=SAEVariant.TOPK,
        top_k=(5,5,5,5,5),
    ),
     SAEConfig(
        name="top20.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 8, 8, 8, 8)),
        sae_variant=SAEVariant.TOPK,
        top_k=(20,20,20,20,20),
    ),
    SAEConfig(
        name="topk-x40.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (40, 40, 40, 40, 40)),
        sae_variant=SAEVariant.TOPK,
        top_k=(10,10,10,10,10),
    ),
    SAEConfig(
        name="jumprelu-x8.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 8, 8, 8, 8)),
        sae_variant=SAEVariant.JUMP_RELU,
    ),
    SAEConfig(
        name="jumprelu-staircase-x8.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 16, 24, 32, 40)),
        sae_variant=SAEVariant.JUMP_RELU_STAIRCASE,
    ),
    SAEConfig(
        name="jumprelu-x16.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        # Only the penultimate layer needs the full x16 expansion factor
        n_features=tuple(64 * n for n in (4, 4, 4, 16, 4)),
        sae_variant=SAEVariant.JUMP_RELU,
    ),
    SAEConfig(
        name="jumprelu-x32.stories_256x4",
        gpt_config=gpt_options["tiktoken_256x4"],
        n_features=tuple(256 * n for n in (32, 32, 32, 32, 32)),
        sae_variant=SAEVariant.JUMP_RELU,
    ),
)
