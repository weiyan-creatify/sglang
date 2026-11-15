# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Pipeline stages for diffusion models.

This package contains the various stages that can be composed to create
complete diffusion pipelines.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
    from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
        CausalDMDDenoisingStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.comfyui_latent_preparation import (
        ComfyUILatentPreparationStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.condition_encoding import (
        AuxiliaryConditionEncodingStage,
        ConditionEncodingStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
        DecodingStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
        DenoisingStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising_dmd import (
        DmdDenoisingStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
        ImageEncodingStage,
        ImageVAEEncodingStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
        InputValidationStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
        LatentPreparationStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
        TextEncodingStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation import (
        DMDTimestepPreparationStage,
        TimestepPreparationStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.vla import (
        VLAActionDenoisingStage,
        VLAActionPostprocessStage,
        VLAObservationPreprocessStage,
        VLAPrefixEncodingStage,
    )

# Each stage imports its loaders and models; importing one stage module (or a
# config importing stage constants) must not import all of them.
_EXPORTS = {
    "PipelineStage": "sglang.multimodal_gen.runtime.pipelines_core.stages.base",
    "CausalDMDDenoisingStage": "sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising",
    "ComfyUILatentPreparationStage": "sglang.multimodal_gen.runtime.pipelines_core.stages.comfyui_latent_preparation",
    "AuxiliaryConditionEncodingStage": "sglang.multimodal_gen.runtime.pipelines_core.stages.condition_encoding",
    "ConditionEncodingStage": "sglang.multimodal_gen.runtime.pipelines_core.stages.condition_encoding",
    "DecodingStage": "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding",
    "DenoisingStage": "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising",
    "DmdDenoisingStage": "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising_dmd",
    "ImageEncodingStage": "sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding",
    "ImageVAEEncodingStage": "sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding",
    "InputValidationStage": "sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation",
    "LatentPreparationStage": "sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation",
    "TextEncodingStage": "sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding",
    "DMDTimestepPreparationStage": "sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation",
    "TimestepPreparationStage": "sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation",
    "VLAActionDenoisingStage": "sglang.multimodal_gen.runtime.pipelines_core.stages.vla",
    "VLAActionPostprocessStage": "sglang.multimodal_gen.runtime.pipelines_core.stages.vla",
    "VLAObservationPreprocessStage": "sglang.multimodal_gen.runtime.pipelines_core.stages.vla",
    "VLAPrefixEncodingStage": "sglang.multimodal_gen.runtime.pipelines_core.stages.vla",
}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "PipelineStage",
    "InputValidationStage",
    "TimestepPreparationStage",
    "DMDTimestepPreparationStage",
    "LatentPreparationStage",
    "ComfyUILatentPreparationStage",
    "DenoisingStage",
    "DmdDenoisingStage",
    "CausalDMDDenoisingStage",
    "ConditionEncodingStage",
    "AuxiliaryConditionEncodingStage",
    "DecodingStage",
    "ImageEncodingStage",
    "ImageVAEEncodingStage",
    "TextEncodingStage",
    "VLAObservationPreprocessStage",
    "VLAPrefixEncodingStage",
    "VLAActionDenoisingStage",
    "VLAActionPostprocessStage",
]
