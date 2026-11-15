# SPDX-License-Identifier: Apache-2.0

"""MiniMax H3-specific pipeline stages."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .stages.audio_encoding import MiniMaxH3AudioEncodingStage
    from .stages.comfyui_step import MiniMaxH3ComfyUIStepStage
    from .stages.decoding import MiniMaxH3DecodingStage
    from .stages.denoising import MiniMaxH3DenoisingStage
    from .stages.latent_preparation import MiniMaxH3LatentPreparationStage
    from .stages.text_encoding import MiniMaxH3TextEncodingStage
    from .stages.timestep_preparation import MiniMaxH3TimestepPreparationStage
    from .stages.visual_encoding import MiniMaxH3VisualEncodingStage

# The stages import the H3 models; constants and plan modules in this package
# are imported by configs that must not pay for them.
_EXPORTS = {
    "MiniMaxH3AudioEncodingStage": ".stages.audio_encoding",
    "MiniMaxH3ComfyUIStepStage": ".stages.comfyui_step",
    "MiniMaxH3DecodingStage": ".stages.decoding",
    "MiniMaxH3DenoisingStage": ".stages.denoising",
    "MiniMaxH3LatentPreparationStage": ".stages.latent_preparation",
    "MiniMaxH3TextEncodingStage": ".stages.text_encoding",
    "MiniMaxH3TimestepPreparationStage": ".stages.timestep_preparation",
    "MiniMaxH3VisualEncodingStage": ".stages.visual_encoding",
}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "MiniMaxH3AudioEncodingStage",
    "MiniMaxH3ComfyUIStepStage",
    "MiniMaxH3DecodingStage",
    "MiniMaxH3DenoisingStage",
    "MiniMaxH3LatentPreparationStage",
    "MiniMaxH3TextEncodingStage",
    "MiniMaxH3TimestepPreparationStage",
    "MiniMaxH3VisualEncodingStage",
]
