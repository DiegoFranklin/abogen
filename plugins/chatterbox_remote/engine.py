"""Chatterbox-remote Engine adapter for the TTS Plugin Architecture.

Unlike kokoro/supertonic, this engine runs no model locally. It forwards
synthesis requests to the gpu-broker (~/Projects/homelab-infra/stacks/gpu-broker/,
Vikunja [125]), which queues the job, wakes the GPU desktop if needed via
desktop-controller, and dispatches it to chatterbox-gpu
(~/Projects/homelab-infra/src/chatterbox-gpu/, Vikunja [124]) -- the resident
service that actually runs the Chatterbox Multilingual pt-br finetune.

`context.http_client` (HostContext) is NOT used here: as of this writing both
production call sites that build HostContext (abogen/tts_plugin/utils.py)
pass a stub whose get()/post() always return None -- there is no real HTTP
client behind it. This plugin brings its own `httpx` dependency instead.
"""
from __future__ import annotations

import base64
import io
import logging
import wave
from typing import Any

import httpx
import numpy as np

from abogen.domain.enums import Language
from abogen.tts_plugin.errors import EngineError, NetworkError
from abogen.tts_plugin.types import (
    AudioFormat,
    Duration,
    SynthesisRequest,
    SynthesizedAudio,
)

logger = logging.getLogger(__name__)

# Only the pt-br finetune is wired up on the broker side today (chatterbox-gpu
# only bundles the bm_george reference clip, voice cloned for pt-br -- see ADR
# addendum 6 of epub-audiobook). Stricter than kokoro's silent EN_US fallback,
# matching supertonic's pattern of raising on anything unsupported.
_SUPPORTED_LANGUAGES = {Language.PT_BR: "pt"}

# Defaults validated and approved by the user, ADR addendum 6.
_DEFAULT_EXAGGERATION = 0.65
_DEFAULT_CFG_WEIGHT = 0.5
_DEFAULT_TEMPERATURE = 0.8
_DEFAULT_VOICE = "bm_george"  # reference_audio/<voice>.wav on chatterbox-gpu, not an abogen voice id

_JOB_TYPE = "chatterbox-pt-br"
# chatterbox-gpu splits any oversized chunk into <=400-char pieces and
# synthesizes them sequentially within one /synthesize call (see its
# server.py) - real chunks from this pipeline have been observed up to
# ~4700 chars (abogen's own chunker not always honoring "Sentence"
# granularity; out of scope to fix here), which can take 17+ sequential
# generations, ~150s+ wall clock for ONE request. 120s was tuned for a
# single normal-length chunk and isn't enough margin for that case -
# confirmed in production: real job timed out mid-book on exactly this.
_REQUEST_TIMEOUT_SECONDS = 300.0
_HTTP_TIMEOUT_SECONDS = _REQUEST_TIMEOUT_SECONDS + 60.0  # headroom over the broker's own wake/queue wait


def supported_languages() -> list[Language]:
    return list(_SUPPORTED_LANGUAGES.keys())


def engine_language(lang: Language) -> str:
    result = _SUPPORTED_LANGUAGES.get(lang)
    if result is None:
        raise ValueError(
            f"chatterbox_remote does not support language: {lang!r}. "
            f"Supported: {supported_languages()}"
        )
    return result


def _pcm_float32_and_duration(wav_bytes: bytes) -> tuple[bytes, float]:
    """Decode a WAV file into raw float32 PCM bytes + duration.

    abogen/tts_plugin/utils.py's Pipeline.__call__ does
    `np.frombuffer(result.data, dtype=np.float32)` directly on
    SynthesizedAudio.data -- despite the AudioFormat declaring "audio/wav",
    the host actually expects headerless float32 PCM, not a real WAV file.
    kokoro/supertonic's engines happen to satisfy this by writing WAV as
    32-bit float samples (sf.write defaults to float for a float32 ndarray),
    so their 44-byte header (a multiple of 4) just becomes ~11 inaudible
    garbage samples prepended -- never raises, nobody noticed. chatterbox-gpu
    writes 16-bit PCM instead (torchaudio's wav default for a float tensor),
    whose header+body size isn't reliably a multiple of 4, which throws
    "buffer size must be a multiple of element size" instead of silently
    corrupting a few samples. Parse properly instead of relying on either
    coincidence.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        raw_frames = wav_file.readframes(frames)

    if sample_width == 2:
        samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw_frames, dtype=np.float32)
    else:
        raise EngineError(f"unexpected WAV sample width from chatterbox-gpu: {sample_width} bytes")

    duration_seconds = frames / rate if rate else 0.0
    return samples.tobytes(), duration_seconds


class ChatterboxRemoteSession:
    """EngineSession implementation that forwards synthesis to the gpu-broker.

    Owns an httpx.Client so repeated synthesize() calls (one per abogen text
    chunk, dozens-to-hundreds per book) reuse the same TCP connection to the
    broker instead of reconnecting every time.
    """

    def __init__(self, broker_url: str) -> None:
        self._broker_url = broker_url
        self._client = httpx.Client(timeout=_HTTP_TIMEOUT_SECONDS)
        self._disposed = False

    def synthesize(self, request: SynthesisRequest) -> SynthesizedAudio:
        if self._disposed:
            raise EngineError("Session disposed")

        params = request.parameters.values
        payload = {
            "text": request.text,
            "language_id": "pt",
            "exaggeration": float(params.get("exaggeration", _DEFAULT_EXAGGERATION)),
            "cfg_weight": float(params.get("cfg_weight", _DEFAULT_CFG_WEIGHT)),
            "temperature": float(params.get("temperature", _DEFAULT_TEMPERATURE)),
            "voice": _DEFAULT_VOICE,
        }

        try:
            response = self._client.post(
                f"{self._broker_url}/jobs",
                json={
                    "job_type": _JOB_TYPE,
                    "payload": payload,
                    "timeout_seconds": _REQUEST_TIMEOUT_SECONDS,
                },
            )
        except httpx.HTTPError as exc:
            raise NetworkError(f"gpu-broker request failed: {exc}") from exc

        try:
            data: dict[str, Any] = response.json()
        except ValueError as exc:
            raise EngineError(f"gpu-broker returned a non-JSON response: {response.text[:300]}") from exc

        if not data.get("ok"):
            raise EngineError(f"gpu-broker job failed: {data.get('error')}")

        result = data["result"]
        wav_bytes = base64.b64decode(result["audio_base64"])
        pcm_float32, duration_seconds = _pcm_float32_and_duration(wav_bytes)

        return SynthesizedAudio(
            data=pcm_float32,
            format=AudioFormat(mime="audio/wav", extension="wav"),
            duration=Duration(seconds=duration_seconds),
        )

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._client.close()


class ChatterboxRemoteEngine:
    """Engine implementation for chatterbox_remote.

    Factory for ChatterboxRemoteSession instances. Stateless itself -- there is
    no local model to hold, just the broker URL.
    """

    def __init__(self, broker_url: str) -> None:
        self._broker_url = broker_url
        self._disposed = False

    def createSession(self) -> ChatterboxRemoteSession:
        if self._disposed:
            raise EngineError("Engine disposed")
        return ChatterboxRemoteSession(self._broker_url)

    def dispose(self) -> None:
        self._disposed = True
