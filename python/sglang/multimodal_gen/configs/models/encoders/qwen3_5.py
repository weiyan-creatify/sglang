# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sglang.multimodal_gen.configs.models.encoders.base import (
    TextEncoderArchConfig,
    TextEncoderConfig,
)

if TYPE_CHECKING:
    from transformers import Qwen3_5TextConfig


def _default_text_config() -> Qwen3_5TextConfig:
    # transformers costs seconds to import; this module loads with the registry.
    from transformers import Qwen3_5TextConfig

    return Qwen3_5TextConfig()


@dataclass
class Qwen3_5EncoderArchConfig(TextEncoderArchConfig):
    architectures: list[str] = field(default_factory=lambda: ["Qwen3_5Model"])
    text_config: Qwen3_5TextConfig | dict = field(default_factory=_default_text_config)


@dataclass
class Qwen3_5EncoderConfig(TextEncoderConfig):
    arch_config: Qwen3_5EncoderArchConfig = field(
        default_factory=Qwen3_5EncoderArchConfig
    )

    def post_diffusers_config_update(self):
        text = self.arch_config.text_config
        if isinstance(text, dict):
            from transformers import Qwen3_5TextConfig

            text = Qwen3_5TextConfig(**text)
            self.arch_config.text_config = text
        self.arch_config.hidden_size = text.hidden_size
        self.arch_config.num_hidden_layers = text.num_hidden_layers
        self.arch_config.num_attention_heads = text.num_attention_heads
        self.arch_config.intermediate_size = text.intermediate_size
        self.arch_config.vocab_size = text.vocab_size
