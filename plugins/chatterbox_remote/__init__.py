"""Chatterbox-remote TTS Plugin for the TTS Plugin Architecture.

Forwards synthesis to the homelab's gpu-broker (Vikunja [125]) instead of
running a model in this process. See engine.py for why.

Exports:
    - PLUGIN_MANIFEST: PluginManifest
    - MODEL_REQUIREMENTS: list[ModelManifest]
    - create_engine: Factory function
"""
from __future__ import annotations

import os
from pathlib import Path

from abogen.tts_plugin.engine import Engine
from abogen.tts_plugin.errors import EngineError
from abogen.tts_plugin.host_context import HostContext
from abogen.tts_plugin.manifest import (
    AudioFormatManifest,
    EngineManifest,
    ModelManifest,
    ParameterManifest,
    PluginManifest,
    RequirementManifest,
    VoiceManifest,
    VoiceSourceManifest,
)
from abogen.tts_plugin.types import EngineConfig

from .engine import ChatterboxRemoteEngine, engine_language

DEFAULT_BROKER_URL = "http://100.109.72.2:8100"

PLUGIN_MANIFEST = PluginManifest(
    id="chatterbox_remote",
    name="Chatterbox (remoto, GPU sob demanda)",
    version="0.1.0",
    api_version="1.0",
    description=(
        "Chatterbox Multilingual (finetune pt-br, voice cloning cross-lingual do "
        "bm_george/Kokoro) rodando num desktop com GPU separado, via gpu-broker do homelab."
    ),
    author="homelab",
    capabilities=(),
    requires=RequirementManifest(internet=True),
    engine=EngineManifest(
        voiceSources=(
            VoiceSourceManifest(
                id="builtin",
                name="Vozes fixas",
                type="list",
                config={"voices": "catálogo estático, ver PLUGIN_MANIFEST.voices"},
            ),
        ),
        parameters=(
            ParameterManifest(
                id="exaggeration",
                name="Exaggeration",
                description="Expressividade da narração (ADR addendum 6: 0.65 validado)",
                type="float",
                default=0.65,
                min=0.0,
                max=2.0,
                step=0.05,
            ),
            ParameterManifest(
                id="cfg_weight",
                name="CFG weight",
                description="Peso de classifier-free guidance (ADR addendum 6: 0.5 validado)",
                type="float",
                default=0.5,
                min=0.0,
                max=1.0,
                step=0.05,
            ),
            ParameterManifest(
                id="temperature",
                name="Temperature",
                description="Temperatura de amostragem (ADR addendum 6: 0.8 validado)",
                type="float",
                default=0.8,
                min=0.0,
                max=2.0,
                step=0.05,
            ),
        ),
        audioFormats=(AudioFormatManifest(mime="audio/wav", extension="wav"),),
    ),
    voices=(
        VoiceManifest(
            id="chatterbox_ptbr_george",
            name="Narrador PT-BR (Chatterbox, clone bm_george)",
            tags=("pt-br", "male", "clone"),
        ),
    ),
)

MODEL_REQUIREMENTS: list[ModelManifest] = []  # no local model -- runs on the remote GPU box


def create_engine(
    context: HostContext,
    model_path: Path | None,
    config: EngineConfig,
) -> Engine:
    """Create a chatterbox_remote engine instance.

    Args:
        context: Host services -- NOT used here (http_client is a dead stub
            as of this writing, see engine.py docstring). Kept in the
            signature to satisfy the plugin contract.
        model_path: Ignored -- there is no local model.
        config: Engine initialization settings. config.language must be
            Language.PT_BR; raises EngineError otherwise (no silent fallback).

    Returns:
        A ChatterboxRemoteEngine instance.

    Raises:
        EngineError: If config.language is unsupported.
    """
    try:
        engine_language(config.language)
    except ValueError as e:
        raise EngineError(str(e)) from e

    broker_url = os.environ.get("GPU_BROKER_URL", DEFAULT_BROKER_URL)
    return ChatterboxRemoteEngine(broker_url)
