"""CLI entry point.

Commands
--------
  movier transcribe  — extract audio + run STT → word timestamps JSON
  movier subtitles   — generate SRT from word timestamps (or video directly)
  movier refine      — clean filler words and join mid-sentence segments
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
    return {"google": tts.synthesize_google}[backend]


def _resolve_tts_timed(backend: str):
    from demo_movier import tts
    return {"google": tts.synthesize_google_timed}[backend]


def _resolve_tts_timed_with_silences(backend: str):
    from demo_movier import tts
    return {"google": tts.synthesize_google_timed_respect_silences}[backend]


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
# refine                                                                        #
# --------------------------------------------------------------------------- #

@cli.command()
@click.argument("srt")
@click.option("--backend", default="rules", show_default=True,
              type=click.Choice(["rules", "llm"]),
              help="'rules' is offline; 'llm' uses Gemini Flash via Vertex AI "
                   "(reuses GOOGLE_CLOUD_PROJECT + ADC, install: uv sync --extra llm).")
@click.option("--model", default="gemini-2.0-flash", show_default=True,
              help="Gemini model to use with --backend llm.")
@click.option("--join-pause", default=0.8, show_default=True,
              help="(rules) Max silence gap in seconds between segments to consider joining.")
@click.option("-o", "--output", default=None,
              help="Output SRT path (default: <srt>.refined.srt).")
def refine(srt, backend, model, join_pause, output):
    """Remove filler words and join mid-sentence subtitle segments.

    Produces a cleaner SRT suited for TTS synthesis. Use the refined SRT
    with 'movier voice' for more natural-sounding narration.

    \b
    Typical workflow:
      movier subtitles video.mp4          # → video.srt
      movier refine video.srt             # → video.refined.srt
      movier voice video.refined.srt --video video.mp4
    """
    from demo_movier import refine as ref
    from demo_movier.subtitles import load_srt, to_srt

    output = output or _stem(srt.removesuffix(".srt"), "") + ".refined.srt"
    subs = load_srt(srt)
    before = len(subs)

    if backend == "rules":
        refined = ref.refine_rules(subs, join_pause_threshold=join_pause)
    else:
        refined = ref.refine_llm(subs, model=model)

    Path(output).write_text(to_srt(refined), encoding="utf-8")
    click.echo(f"  {before} → {len(refined)} subtitles → {output}")


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
@click.option("--tts", default="google", type=click.Choice(["google"]),
              show_default=True)
@click.option("--voice", default="en-US-Chirp3-HD-Charon", show_default=True,
              help="Voice name. See tts.GOOGLE_VOICES for available options.")
@click.option("--not-timed", is_flag=True,
              help="Ignore subtitle timestamps")
@click.option("--video", default=None, help="Source video (needed for timed duration).")
@click.option("--rate", default=1.0, show_default=True, help="Speaking rate (0.5–2.0).")
@click.option("-o", "--output", default=None)
@click.option("--respect-silences", is_flag=True)
def voice(srt_or_text, tts, voice, not_timed, video, rate, output, respect_silences):
    """Synthesise a TTS audio file from an SRT or plain text file."""
    from demo_movier import tts as tts_mod, subtitles, video as vid

    output = output or _stem(srt_or_text, ".tts.mp3")
    timed = not not_timed

    if srt_or_text.endswith(".srt"):
        subs = subtitles.load_srt(srt_or_text)
        full_text = " ".join(s.text for s in subs)
    else:
        full_text = Path(srt_or_text).read_text(encoding="utf-8")
        subs = []

    if timed and subs:
        if not video:
            raise click.UsageError("--video is required for timed results")
        duration = vid.video_duration(video)
        click.echo(f"  TTS (timed, {tts.upper()}, {len(subs)} segments) …")
        if respect_silences:
            _resolve_tts_timed_with_silences(tts)(subs, duration, output, voice_name=voice)
        else:
            _resolve_tts_timed(tts)(subs, duration, output, voice_name=voice)
    else:
        click.echo(f"  TTS (full, {tts.upper()}) …")
        fn = _resolve_tts_full(tts)
        fn(full_text, output, voice_name=voice, speaking_rate=rate)

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
@click.option("--tts", default="google", type=click.Choice(["google", "none"]),
              show_default=True, help="Pass 'none' to skip voice replacement.")
@click.option("--voice", default="en-US-Chirp3-HD-Charon", show_default=True)
@click.option("--language", default="en-US", show_default=True)
@click.option("--max-words", default=8, show_default=True)
@click.option("--font-size", default=22, show_default=True)
@click.option("--color", default="white",
              type=click.Choice(["white", "yellow", "cyan"]), show_default=True)
@click.option("--not-timed", is_flag=True,
              help="Ignore subtitle timing timing.")
@click.option("--rate", default=1.0, show_default=True)
@click.option("--refine-backend", default="llm", show_default=True,
              type=click.Choice(["rules", "llm"]),
              help="Subtitle refinement backend: 'rules' is offline, 'llm' uses Gemini Flash.")
@click.option("--resume", is_flag=True,
              help="Skip any step whose output file already exists, resuming from where a previous run stopped.")
@click.option("--keep-intermediates", is_flag=True,
              help="Keep .words.json, .srt, .refined.srt, .tts.mp3 files alongside the output.")
@click.option("--no-burn-subtitles", is_flag=True,
              help="Skip burning subtitles into the final video. With --tts=none, only SRT files are produced.")
@click.option("-o", "--output", default=None,
              help="Final output video path (default: <video>.final.mp4).")
@click.option("--respect-silences", is_flag=True)
def run(video, stt, tts, voice, language, max_words, font_size, color, not_timed,
        rate, refine_backend, resume, keep_intermediates, no_burn_subtitles, output, respect_silences):
    """Full pipeline: transcribe → SRT → refine → TTS → extract subtitles → burn → replace audio."""
    from demo_movier import stt as stt_mod, subtitles, video as vid
    from demo_movier import refine as ref

    timed = not not_timed
    base = _stem(video, "")
    output = output or _stem(video, ".final.mp4")

    color_map = {
        "white":  ("&H00FFFFFF", "&H00000000"),
        "yellow": ("&H0000FFFF", "&H00000000"),
        "cyan":   ("&H00FFFF00", "&H00000000"),
    }
    primary, outline = color_map[color]
    # resume requires intermediates on disk so steps can be detected as complete
    save_intermediates = keep_intermediates or resume

    with tempfile.TemporaryDirectory() as tmp:
        wav          = os.path.join(tmp, "audio.wav")
        srt_path     = f"{base}.srt"
        refined_srt  = f"{base}.refined.srt"
        words_json   = f"{base}.words.json"
        tts_audio    = f"{base}.tts.mp3"
        tts_wav      = os.path.join(tmp, "tts.wav")
        tts_srt      = f"{base}.tts.srt"
        subbed       = os.path.join(tmp, "subtitled.mp4")

        total = 4 if tts == "none" else (6 if no_burn_subtitles else 7)

        # 1+2. Extract audio + Transcribe (coupled: both skipped when words JSON exists)
        if resume and Path(words_json).exists():
            click.echo(f"[1/{total}] Skipping audio extraction (words JSON exists) …")
            click.echo(f"[2/{total}] Skipping transcription — loading {words_json} …")
            raw = json.loads(Path(words_json).read_text())
            from demo_movier.subtitles import Word
            words = [Word(text=w["word"], start=w["start"], end=w["end"]) for w in raw]
        else:
            click.echo(f"[1/{total}] Extracting audio …")
            vid.extract_audio(video, wav)
            click.echo(f"[2/{total}] Transcribing with {stt.upper()} …")
            if stt == "google":
                words = stt_mod.transcribe_google(wav, language=language)
            else:
                words = stt_mod.transcribe_whisper(wav)
            if save_intermediates:
                data = [{"word": w.text, "start": w.start, "end": w.end} for w in words]
                Path(words_json).write_text(json.dumps(data, indent=2, ensure_ascii=False))
                if keep_intermediates:
                    click.echo(f"      saved {words_json}")

        # 3. Generate SRT
        if resume and Path(srt_path).exists():
            click.echo(f"[3/{total}] Skipping subtitle generation — loading {srt_path} …")
            subs = subtitles.load_srt(srt_path)
        else:
            click.echo(f"[3/{total}] Generating subtitles …")
            subs = subtitles.group_into_subtitles(words, max_words=max_words)
            Path(srt_path).write_text(subtitles.to_srt(subs), encoding="utf-8")
            if keep_intermediates:
                click.echo(f"      saved {srt_path}")

        # 4. Refine SRT
        if resume and Path(refined_srt).exists():
            click.echo(f"[4/{total}] Skipping refinement — loading {refined_srt} …")
            refined_subs = subtitles.load_srt(refined_srt)
        else:
            click.echo(f"[4/{total}] Refining subtitles ({refine_backend}) …")
            if refine_backend == "rules":
                refined_subs = ref.refine_rules(subs)
            else:
                refined_subs = ref.refine_llm(subs)
            Path(refined_srt).write_text(subtitles.to_srt(refined_subs), encoding="utf-8")
            if keep_intermediates:
                click.echo(f"      saved {refined_srt}")

        if tts == "none":
            if no_burn_subtitles:
                click.echo(f"\nDone — subtitles saved to {refined_srt}")
            else:
                if resume and Path(output).exists():
                    click.echo(f"  Skipping subtitle burn — {output} already exists …")
                else:
                    click.echo(f"  Burning subtitles …")
                    vid.burn_subtitles(video, refined_srt, output,
                                       font_size=font_size,
                                       primary_color=primary, outline_color=outline)
                click.echo(f"\nDone → {output}")
            return

        # 5. TTS (from refined subtitles)
        if resume and Path(tts_audio).exists():
            click.echo(f"[5/{total}] Skipping TTS — {tts_audio} already exists …")
        else:
            click.echo(f"[5/{total}] Synthesising voice ({tts.upper()}) …")
            duration = vid.video_duration(video)
            if timed:
                if respect_silences:
                    _resolve_tts_timed_with_silences(tts)(refined_subs, duration, tts_audio, voice_name=voice)
                else:
                    _resolve_tts_timed(tts)(refined_subs, duration, tts_audio, voice_name=voice)
            else:
                full_text = " ".join(s.text for s in refined_subs)
                fn = _resolve_tts_full(tts)
                fn(full_text, tts_audio, voice_name=voice, speaking_rate=rate)
            if keep_intermediates:
                click.echo(f"      saved {tts_audio}")

        if no_burn_subtitles:
            # 6. Replace audio directly (no subtitle burn)
            if resume and Path(output).exists():
                click.echo(f"[6/{total}] Skipping — {output} already exists …")
            else:
                click.echo(f"[6/{total}] Replacing audio …")
                vid.replace_audio(video, tts_audio, output)
        else:
            # 6. Extract subtitles from TTS audio (re-transcribe to get accurate timings)
            if resume and Path(tts_srt).exists():
                click.echo(f"[6/{total}] Skipping TTS transcription — {tts_srt} already exists …")
            else:
                click.echo(f"[6/{total}] Extracting subtitles from TTS audio …")
                vid.extract_audio(tts_audio, tts_wav)
                if stt == "google":
                    tts_words = stt_mod.transcribe_google(tts_wav, language=language)
                else:
                    tts_words = stt_mod.transcribe_whisper(tts_wav)
                tts_subs = subtitles.group_into_subtitles(tts_words, max_words=max_words)
                Path(tts_srt).write_text(subtitles.to_srt(tts_subs), encoding="utf-8")
                if keep_intermediates:
                    click.echo(f"      saved {tts_srt}")

            # 7. Burn subtitles (from TTS transcription) then replace audio
            if resume and Path(output).exists():
                click.echo(f"[7/{total}] Skipping — {output} already exists …")
            else:
                click.echo(f"[7/{total}] Burning subtitles …")
                vid.burn_subtitles(video, tts_srt, subbed,
                                   font_size=font_size,
                                   primary_color=primary, outline_color=outline)
                vid.replace_audio(subbed, tts_audio, output)

    if not save_intermediates:
        for f in [srt_path, refined_srt, words_json, tts_audio, tts_srt]:
            Path(f).unlink(missing_ok=True)

    click.echo(f"\nDone → {output}")
