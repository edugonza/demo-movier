"""Text-to-Speech backends.

Supported:
  google     — Google Cloud TTS with Studio / Neural2 / Chirp3 HD voices.
               Studio voices (en-US-Studio-Q/O) are the highest quality.
               Chirp3 HD voices (e.g. en-US-Chirp3-HD-Aoede) are the newest
               and sound the most natural.
  elevenlabs — ElevenLabs API (best overall voice quality for demos).
               Requires: ELEVENLABS_API_KEY env var.
               Install extras: uv sync --extra elevenlabs

Two synthesis modes:
  full    — Synthesise the entire transcript as one audio file.
            Simple, but the pacing won't match the original video.
  timed   — Synthesise each subtitle segment independently and stitch them
            together with silence, preserving the original timing layout.
            Requires pydub.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

from demo_movier.subtitles import Subtitle


# ---------------------------------------------------------------------------
# Google Cloud Text-to-Speech
# ---------------------------------------------------------------------------

# Best voices ranked by quality (2025):
#   Chirp3 HD  — ultra-natural, multi-speaker, newest model
#   Studio     — very natural, designed for long-form narration
#   Neural2    — good quality, wide language support
GOOGLE_VOICES = {
    "studio-male":      "en-US-Studio-Q",
    "studio-female":    "en-US-Studio-O",
    "chirp3-aoede":     "en-US-Chirp3-HD-Aoede",   # warm female
    "chirp3-charon":    "en-US-Chirp3-HD-Charon",  # deep male
    "chirp3-fenrir":    "en-US-Chirp3-HD-Fenrir",  # authoritative male
    "chirp3-kore":      "en-US-Chirp3-HD-Kore",    # clear female
}


def synthesize_google(
    text: str,
    output_path: str,
    voice_name: str = "en-US-Studio-Q",
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
) -> None:
    """Synthesise full text to a single MP3 file."""
    from google.cloud import texttospeech

    client = texttospeech.TextToSpeechClient()
    lang = "-".join(voice_name.split("-")[:2])  # e.g. "en-US"

    response = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=texttospeech.VoiceSelectionParams(
            language_code=lang,
            name=voice_name,
        ),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate,
            pitch=pitch,
        ),
    )
    Path(output_path).write_bytes(response.audio_content)


def synthesize_google_timed(
    subtitles: list[Subtitle],
    total_duration: float,
    output_path: str,
    voice_name: str = "en-US-Studio-Q",
    speaking_rate: float = 1.0,
) -> None:
    """Synthesise each subtitle and overlay them at their original timestamps.

    The result is an audio file with the same total duration as the video,
    where each TTS clip starts at the subtitle's original start time.
    Clips that run longer than their subtitle slot will overlap with the next
    one — reduce speaking_rate or use shorter subtitles to avoid this.
    """
    from google.cloud import texttospeech
    from pydub import AudioSegment  # type: ignore

    client = texttospeech.TextToSpeechClient()
    lang = "-".join(voice_name.split("-")[:2])

    base = AudioSegment.silent(duration=int(total_duration * 1000))

    for sub in subtitles:
        response = client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=sub.text),
            voice=texttospeech.VoiceSelectionParams(
                language_code=lang,
                name=voice_name,
            ),
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=speaking_rate,
            ),
        )
        clip = AudioSegment.from_mp3(io.BytesIO(response.audio_content))
        base = base.overlay(clip, position=int(sub.start * 1000))

    base.export(output_path, format="mp3")


# ---------------------------------------------------------------------------
# ElevenLabs
# ---------------------------------------------------------------------------

ELEVENLABS_VOICES = {
    # A few popular presets — list all with: elevenlabs.voices()
    "rachel":  "21m00Tcm4TlvDq8ikWAM",
    "adam":    "pNInz6obpgDQGcFmaJgB",
    "bella":   "EXAVITQu4vr4xnSDxMaL",
    "josh":    "TxGEqnHWrfWFTfGW9XjX",
}


def synthesize_elevenlabs(
    text: str,
    output_path: str,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel
    model_id: str = "eleven_multilingual_v2",
    stability: float = 0.5,
    similarity_boost: float = 0.75,
) -> None:
    """Synthesise text with ElevenLabs and save as MP3."""
    from elevenlabs import ElevenLabs, VoiceSettings  # type: ignore

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    client = ElevenLabs(api_key=api_key)

    audio = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=model_id,
        voice_settings=VoiceSettings(
            stability=stability,
            similarity_boost=similarity_boost,
        ),
        output_format="mp3_44100_128",
    )
    with open(output_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)


def synthesize_elevenlabs_timed(
    subtitles: list[Subtitle],
    total_duration: float,
    output_path: str,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    model_id: str = "eleven_multilingual_v2",
) -> None:
    """ElevenLabs timed synthesis — same overlay strategy as Google timed."""
    from elevenlabs import ElevenLabs, VoiceSettings  # type: ignore
    from pydub import AudioSegment  # type: ignore

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    client = ElevenLabs(api_key=api_key)
    base = AudioSegment.silent(duration=int(total_duration * 1000))

    for sub in subtitles:
        audio_bytes = b"".join(client.text_to_speech.convert(
            text=sub.text,
            voice_id=voice_id,
            model_id=model_id,
            voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.75),
            output_format="mp3_44100_128",
        ))
        clip = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
        base = base.overlay(clip, position=int(sub.start * 1000))

    base.export(output_path, format="mp3")
