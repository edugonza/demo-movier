"""CLI entry point.

Commands
--------
  movier transcribe  — extract audio + run STT → word timestamps JSON
  movier subtitles   — generate SRT from word timestamps (or video directly)
  movier burn        — hardcode subtitles into video
  movier voice       — synthesise TTS audio from SRT
  movier replace     — replace video audio track
  movier run         — full pipeline in one shot
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _stem(path: str, suffix: str) -> str:
    p = Path(path)
    return str(p.parent / f"{p.stem}{suffix}")


def _resolve_stt(backend: str):
    from demo_movier import stt
    return {"google": stt.transcribe_google, "whisper": stt.transcribe_whisper}[backend]


def _resolve_tts_full(backend: str):
    from demo_movier import tts
    return {"google": tts.synthesize_google, "elevenlabs": tts.synthesize_elevenlabs}[backend]


def _resolve_tts_timed(backend: str):
    from demo_movier import tts
    return {
        "google": tts.synthesize_google_timed,
        "elevenlabs": tts.synthesize_elevenlabs_timed,
    }[backend]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """demo-movier — subtitle + voice replacement pipeline for demo videos."""


# --------------------------------------------------------------------------- #
# transcribe                                                                    #
# --------------------------------------------------------------------------- #

@cli.command()
@click.argument("video")
@click.option("--stt", default="google", type=click.Choice(["google", "whisper"]),
              show_default=True, help="STT backend.")
@click.option("--language", default="en-US", show_default=True)
@click.option("--gcs-uri", default=None,
              help="gs:// URI of audio already on GCS (skips local extraction). "
                   "Required for videos longer than ~60 s with the Google backend.")
@click.option("--model", default="large-v3", show_default=True,
              help="Whisper model size (tiny/base/small/medium/large-v3).")
@click.option("-o", "--output", default=None,
              help="Output JSON path (default: <video>.words.json).")
def transcribe(video, stt, language, gcs_uri, model, output):
    """Extract audio and transcribe to word-level timestamps JSON."""
    from demo_movier import stt as stt_mod, video as vid

    output = output or _stem(video, ".words.json")

    with tempfile.TemporaryDirectory() as tmp:
        if stt == "google" and gcs_uri:
            click.echo(f"  STT: Google Chirp 2 (GCS) ← {gcs_uri}")
            words = stt_mod.transcribe_google_gcs(gcs_uri, language=language)
        else:
            wav = os.path.join(tmp, "audio.wav")
            click.echo(f"  Extracting audio from {video} …")
            vid.extract_audio(video, wav)
            click.echo(f"  STT: {stt.upper()}{' Chirp 2' if stt == 'google' else f' ({model})'} …")
            if stt == "google":
                words = stt_mod.transcribe_google(wav, language=language)
            else:
                words = stt_mod.transcribe_whisper(wav, model_size=model, language=language)

    data = [{"word": w.text, "start": w.start, "end": w.end} for w in words]
    Path(output).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    click.echo(f"  {len(words)} words → {output}")


# --------------------------------------------------------------------------- #
# subtitles                                                                     #
# --------------------------------------------------------------------------- #

@cli.command("subtitles")
@click.argument("source", metavar="VIDEO_OR_WORDS_JSON")
@click.option("--stt", default="google", type=click.Choice(["google", "whisper"]))
@click.option("--language", default="en-US", show_default=True)
@click.option("--max-words", default=8, show_default=True)
@click.option("--max-chars", default=60, show_default=True)
@click.option("--pause", default=0.6, show_default=True,
              help="Start a new subtitle when silence gap exceeds this (seconds).")
@click.option("-o", "--output", default=None, help="Output SRT path.")
def subtitles_cmd(source, stt, language, max_words, max_chars, pause, output):
    """Generate an SRT file from a video or a pre-computed words JSON."""
    from demo_movier import stt as stt_mod, subtitles, video as vid

    output = output or _stem(source.replace(".words.json", ""), ".srt")

    if source.endswith(".json"):
        raw = json.loads(Path(source).read_text())
        from demo_movier.subtitles import Word
        words = [Word(text=w["word"], start=w["start"], end=w["end"]) for w in raw]
    else:
        # Extract + transcribe on the fly
        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "audio.wav")
            click.echo(f"  Extracting audio …")
            vid.extract_audio(source, wav)
            click.echo(f"  STT: {stt.upper()} …")
            fn = _resolve_stt(stt)
            words = fn(wav, language=language) if stt == "google" else fn(wav)

    subs = subtitles.group_into_subtitles(
        words, max_words=max_words, max_chars=max_chars, pause_threshold=pause
    )
    Path(output).write_text(subtitles.to_srt(subs), encoding="utf-8")
    click.echo(f"  {len(subs)} subtitles → {output}")


# --------------------------------------------------------------------------- #
# burn                                                                          #
# --------------------------------------------------------------------------- #

@cli.command()
@click.argument("video")
@click.argument("srt")
@click.option("-o", "--output", default=None)
@click.option("--font", default="Arial", show_default=True)
@click.option("--font-size", default=22, show_default=True)
@click.option("--color", default="white",
              type=click.Choice(["white", "yellow", "cyan"]),
              show_default=True)
@click.option("--no-bold", is_flag=True)
@click.option("--margin-v", default=30, show_default=True,
              help="Bottom margin in pixels.")
def burn(video, srt, output, font, font_size, color, no_bold, margin_v):
    """Hardcode subtitles into a video (creates a new file)."""
    from demo_movier import video as vid

    color_map = {
        "white":  ("&H00FFFFFF", "&H00000000"),
        "yellow": ("&H0000FFFF", "&H00000000"),
        "cyan":   ("&H00FFFF00", "&H00000000"),
    }
    primary, outline = color_map[color]
    output = output or _stem(video, ".subtitled.mp4")

    click.echo(f"  Burning subtitles into {video} …")
    vid.burn_subtitles(
        video, srt, output,
        font_name=font, font_size=font_size,
        primary_color=primary, outline_color=outline,
        bold=not no_bold, margin_v=margin_v,
    )
    click.echo(f"  → {output}")


# --------------------------------------------------------------------------- #
# voice                                                                         #
# --------------------------------------------------------------------------- #

@cli.command()
@click.argument("srt_or_text")
@click.option("--tts", default="google", type=click.Choice(["google", "elevenlabs"]),
              show_default=True)
@click.option("--voice", default="en-US-Studio-Q", show_default=True,
              help="Voice name/ID. Google: see tts.GOOGLE_VOICES. "
                   "ElevenLabs: voice ID or use tts.ELEVENLABS_VOICES presets.")
@click.option("--timed", is_flag=True,
              help="Preserve subtitle timestamps (overlay each clip at its start time). "
                   "Requires --video to determine total duration.")
@click.option("--video", default=None, help="Source video (needed for --timed duration).")
@click.option("--rate", default=1.0, show_default=True, help="Speaking rate (0.5–2.0).")
@click.option("-o", "--output", default=None)
def voice(srt_or_text, tts, voice, timed, video, rate, output):
    """Synthesise a TTS audio file from an SRT or plain text file."""
    from demo_movier import tts as tts_mod, subtitles, video as vid

    output = output or _stem(srt_or_text, ".tts.mp3")

    if srt_or_text.endswith(".srt"):
        subs = subtitles.load_srt(srt_or_text)
        full_text = " ".join(s.text for s in subs)
    else:
        full_text = Path(srt_or_text).read_text(encoding="utf-8")
        subs = []

    if timed and subs:
        if not video:
            raise click.UsageError("--video is required with --timed")
        duration = vid.video_duration(video)
        click.echo(f"  TTS (timed, {tts.upper()}, {len(subs)} segments) …")
        _resolve_tts_timed(tts)(subs, duration, output, voice_name=voice if tts == "google" else voice)
    else:
        click.echo(f"  TTS (full, {tts.upper()}) …")
        fn = _resolve_tts_full(tts)
        if tts == "google":
            fn(full_text, output, voice_name=voice, speaking_rate=rate)
        else:
            fn(full_text, output, voice_id=voice)

    click.echo(f"  → {output}")


# --------------------------------------------------------------------------- #
# replace                                                                       #
# --------------------------------------------------------------------------- #

@cli.command()
@click.argument("video")
@click.argument("audio")
@click.option("-o", "--output", default=None)
@click.option("--mix", is_flag=True, help="Mix new audio with original instead of replacing.")
@click.option("--original-volume", default=0.1, show_default=True,
              help="Original audio volume when mixing (0.0–1.0).")
def replace(video, audio, output, mix, original_volume):
    """Replace (or mix) the audio track in a video."""
    from demo_movier import video as vid

    output = output or _stem(video, ".revoiced.mp4")
    click.echo(f"  Replacing audio in {video} …")
    vid.replace_audio(video, audio, output,
                      keep_original=mix, original_volume=original_volume)
    click.echo(f"  → {output}")


# --------------------------------------------------------------------------- #
# run  (full pipeline)                                                          #
# --------------------------------------------------------------------------- #

@cli.command()
@click.argument("video")
@click.option("--stt", default="google", type=click.Choice(["google", "whisper"]),
              show_default=True)
@click.option("--tts", default="google", type=click.Choice(["google", "elevenlabs"]),
              show_default=True, help="Pass 'none' to skip voice replacement.")
@click.option("--voice", default="en-US-Studio-Q", show_default=True)
@click.option("--language", default="en-US", show_default=True)
@click.option("--max-words", default=8, show_default=True)
@click.option("--font-size", default=22, show_default=True)
@click.option("--color", default="white",
              type=click.Choice(["white", "yellow", "cyan"]), show_default=True)
@click.option("--timed", is_flag=True,
              help="Synthesise TTS per-subtitle to preserve timing.")
@click.option("--rate", default=1.0, show_default=True)
@click.option("--keep-intermediates", is_flag=True,
              help="Keep .words.json, .srt, .tts.mp3 files alongside the output.")
@click.option("-o", "--output", default=None,
              help="Final output video path (default: <video>.final.mp4).")
def run(video, stt, tts, voice, language, max_words, font_size, color, timed,
        rate, keep_intermediates, output):
    """Full pipeline: transcribe → SRT → burn subtitles → TTS → replace audio."""
    from demo_movier import stt as stt_mod, tts as tts_mod, subtitles, video as vid

    base = _stem(video, "")
    output = output or _stem(video, ".final.mp4")

    with tempfile.TemporaryDirectory() as tmp:
        wav       = os.path.join(tmp, "audio.wav")
        srt_path  = f"{base}.srt"
        words_json = f"{base}.words.json"
        tts_audio = f"{base}.tts.mp3"
        subbed    = os.path.join(tmp, "subtitled.mp4")

        # 1. Extract audio
        click.echo("[1/5] Extracting audio …")
        vid.extract_audio(video, wav)

        # 2. Transcribe
        click.echo(f"[2/5] Transcribing with {stt.upper()} …")
        if stt == "google":
            words = stt_mod.transcribe_google(wav, language=language)
        else:
            words = stt_mod.transcribe_whisper(wav)

        if keep_intermediates:
            data = [{"word": w.text, "start": w.start, "end": w.end} for w in words]
            Path(words_json).write_text(json.dumps(data, indent=2, ensure_ascii=False))
            click.echo(f"      saved {words_json}")

        # 3. Generate SRT
        click.echo("[3/5] Generating subtitles …")
        subs = subtitles.group_into_subtitles(words, max_words=max_words)
        Path(srt_path).write_text(subtitles.to_srt(subs), encoding="utf-8")
        if keep_intermediates:
            click.echo(f"      saved {srt_path}")

        # 4. Burn subtitles
        click.echo("[4/5] Burning subtitles …")
        color_map = {
            "white":  ("&H00FFFFFF", "&H00000000"),
            "yellow": ("&H0000FFFF", "&H00000000"),
            "cyan":   ("&H00FFFF00", "&H00000000"),
        }
        primary, outline = color_map[color]

        if tts == "none":
            vid.burn_subtitles(video, srt_path, output,
                               font_size=font_size,
                               primary_color=primary, outline_color=outline)
            click.echo(f"\nDone → {output}")
            return

        vid.burn_subtitles(video, srt_path, subbed,
                           font_size=font_size,
                           primary_color=primary, outline_color=outline)

        # 5. TTS
        click.echo(f"[5/5] Synthesising voice ({tts.upper()}) …")
        duration = vid.video_duration(video)
        if timed:
            _resolve_tts_timed(tts)(subs, duration, tts_audio,
                                    voice_name=voice if tts == "google" else voice)
        else:
            full_text = " ".join(s.text for s in subs)
            fn = _resolve_tts_full(tts)
            if tts == "google":
                fn(full_text, tts_audio, voice_name=voice, speaking_rate=rate)
            else:
                fn(full_text, tts_audio, voice_id=voice)

        if keep_intermediates:
            click.echo(f"      saved {tts_audio}")

        # 6. Replace audio
        vid.replace_audio(subbed, tts_audio, output)

    if not keep_intermediates:
        for f in [srt_path, words_json, tts_audio]:
            Path(f).unlink(missing_ok=True)

    click.echo(f"\nDone → {output}")
