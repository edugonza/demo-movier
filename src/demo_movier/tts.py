"""Text-to-Speech backends.

Supported:
  google     — Google Cloud TTS with Studio / Neural2 / Chirp3 HD voices.
               Studio voices (en-US-Studio-Q/O) are the highest quality.
               Chirp3 HD voices (e.g. en-US-Chirp3-HD-Aoede) are the newest
               and sound the most natural.

Two synthesis modes:
  full    — Synthesise the entire transcript as one audio file.
            Simple, but the pacing won't match the original video.
  timed   — Two-pass SSML approach (Google only).
            Pass 1: single API call with <break> tags for silences; measure
            actual speech duration.
            Pass 2: same SSML but each text segment wrapped in
            <prosody rate="X%"> so the total speech duration matches the
            original video's speech slots.
            Formula: prosody_rate = speech_duration_pass1 / target_speech_duration
            where target_speech_duration = video_duration − total_break_duration.
            Silences are never rescaled, only the spoken parts.
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


_MAX_BYTES = 4500  # conservative — Google's hard limit is 5000 bytes


def synthesize_google(
    text: str,
    output_path: str,
    voice_name: str = "en-US-Studio-Q",
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
) -> None:
    """Synthesise full text to a single MP3 file.

    Uses the standard synchronous API for short text. Automatically falls back
    to the Long Audio API (staging through GCS) when the input exceeds Google's
    5000-byte limit. Set GOOGLE_CLOUD_BUCKET in .env for the long-audio path.
    """
    if len(text.encode()) <= _MAX_BYTES:
        _synthesize_google_short(text, output_path, voice_name, speaking_rate, pitch)
    else:
        _synthesize_google_long(text, output_path, voice_name, speaking_rate, pitch)


def _synthesize_google_short(
    text: str,
    output_path: str,
    voice_name: str,
    speaking_rate: float,
    pitch: float,
) -> None:
    from google.cloud import texttospeech

    project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
    client = texttospeech.TextToSpeechClient(
        client_options={"quota_project_id": project_id}
    )
    lang = "-".join(voice_name.split("-")[:2])
    response = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=texttospeech.VoiceSelectionParams(language_code=lang, name=voice_name),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate,
            pitch=pitch,
        ),
    )
    Path(output_path).write_bytes(response.audio_content)


def _synthesize_google_long(
    text: str,
    output_path: str,
    voice_name: str,
    speaking_rate: float,
    pitch: float,
) -> None:
    """Long Audio API path: writes to GCS then downloads the result."""
    import uuid
    from google.cloud import texttospeech as tts_long
    from google.cloud import storage

    project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
    bucket_name = os.environ.get("GOOGLE_CLOUD_BUCKET")
    if not bucket_name:
        raise RuntimeError(
            "GOOGLE_CLOUD_BUCKET is required when TTS input exceeds 5000 bytes. "
            "Set it in .env to the name of a GCS bucket the service account can write to."
        )

    # Long Audio API only supports LINEAR16; we download the WAV and convert to MP3.
    gcs_key = f"tts-tmp/{uuid.uuid4()}.wav"
    gcs_uri = f"gs://{bucket_name}/{gcs_key}"
    lang = "-".join(voice_name.split("-")[:2])

    client = tts_long.TextToSpeechLongAudioSynthesizeClient(
        client_options={"quota_project_id": project_id}
    )
    request = tts_long.SynthesizeLongAudioRequest(
        parent=f"projects/{project_id}/locations/us-central1",
        input=tts_long.SynthesisInput(text=text),
        voice=tts_long.VoiceSelectionParams(language_code=lang, name=voice_name),
        audio_config=tts_long.AudioConfig(
            audio_encoding=tts_long.AudioEncoding.LINEAR16,
            speaking_rate=speaking_rate,
            pitch=pitch,
        ),
        output_gcs_uri=gcs_uri,
    )

    print(f"  Long Audio API → {gcs_uri} …")
    operation = client.synthesize_long_audio(request=request)
    operation.result(timeout=600)

    from pydub import AudioSegment  # type: ignore
    import tempfile

    blob = storage.Client(project=project_id).bucket(bucket_name).blob(gcs_key)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        blob.download_to_filename(tmp.name)
        blob.delete()
        AudioSegment.from_wav(tmp.name).export(output_path, format="mp3")
    Path(tmp.name).unlink(missing_ok=True)


_SSML_MAX_BYTES = _MAX_BYTES


def synthesize_google_timed_respect_silences(
    subtitles: list[Subtitle],
    total_duration: float,
    output_path: str,
    voice_name: str = "en-US-Studio-Q",
) -> None:
    """Two-pass SSML synthesis that matches the original video's speech pacing.

    When the full SSML exceeds 5000 bytes (Google's hard limit), subtitles are
    split into multiple chunks that are each within the limit.  Both passes
    operate on the same chunks; pass-2 audio is concatenated in order.

    Pass 1 — one API call per chunk, <break> tags only, no prosody.
             Total speech duration = sum(chunk durations) − total_silence.

    Pass 2 — same chunks, every text segment wrapped in <prosody rate="X%">:

        prosody_rate = speech_duration_pass1 / target_speech_duration
        target_speech_duration = total_duration − total_break_duration

    Inter-chunk gaps are preserved as the leading <break> of the next chunk,
    so the silence layout is identical across both passes.
    """
    from google.cloud import texttospeech
    from pydub import AudioSegment  # type: ignore

    project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
    client = texttospeech.TextToSpeechClient(
        client_options={"quota_project_id": project_id}
    )
    lang = "-".join(voice_name.split("-")[:2])
    voice_params = texttospeech.VoiceSelectionParams(language_code=lang, name=voice_name)
    audio_cfg = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    def call(ssml: str) -> bytes:
        return client.synthesize_speech(
            input=texttospeech.SynthesisInput(ssml=ssml),
            voice=voice_params,
            audio_config=audio_cfg,
        ).audio_content

    silences = _timed_silences(subtitles, total_duration)
    total_silence = sum(silences)
    target_speech = total_duration - total_silence  # = sum of subtitle slot widths

    chunks = _ssml_chunks(subtitles, silences)
    if len(chunks) > 1:
        print(f"  SSML too large — split into {len(chunks)} chunks")

    # Pass 1: for each chunk measure actual audio duration, derive per-chunk prosody rate
    prosody_rates: list[float] = []
    total_dur1 = 0.0
    for chunk_subs, chunk_silences in chunks:
        audio = call(_timed_ssml(chunk_subs, chunk_silences, prosody_rate=None))
        chunk_dur = len(AudioSegment.from_mp3(io.BytesIO(audio))) / 1000.0
        total_dur1 += chunk_dur

        chunk_silence = sum(chunk_silences)
        chunk_speech1 = chunk_dur - chunk_silence
        chunk_target_speech = sum(s.end - s.start for s in chunk_subs)
        rate = chunk_speech1 / chunk_target_speech if chunk_target_speech > 0 else 1.0
        prosody_rates.append(rate)

    speech1 = total_dur1 - total_silence
    print(f"  Pass 1: {total_dur1:.2f}s total  |  {speech1:.2f}s speech  |  target {target_speech:.2f}s")
    rates_str = ", ".join(f"{r * 100:.1f}%" for r in prosody_rates)
    print(f"  Pass 2: per-chunk prosody rates [{rates_str}]")

    # Pass 2: apply per-chunk prosody rates and concatenate
    combined = AudioSegment.empty()
    for (chunk_subs, chunk_silences), rate in zip(chunks, prosody_rates):
        audio = call(_timed_ssml(chunk_subs, chunk_silences, prosody_rate=rate))
        segment = AudioSegment.from_mp3(io.BytesIO(audio))

        expected_ms = round((sum(chunk_silences) + sum(s.end - s.start for s in chunk_subs)) * 1000)
        actual_ms = len(segment)
        gap_ms = expected_ms - actual_ms
        if gap_ms > 0:
            print(f"  Pass 2 chunk: padding {gap_ms}ms to match expected {expected_ms}ms (got {actual_ms}ms)")
            segment += AudioSegment.silent(duration=gap_ms)
        elif gap_ms < 0:
            overshoot_ms = -gap_ms
            print(f"  Pass 2 chunk: compressing {overshoot_ms}ms overshoot to fit {expected_ms}ms (got {actual_ms}ms)")
            segment = _rubberband_stretch(segment, actual_ms / expected_ms)

        combined += segment

    combined_ms = len(combined)
    expected_total_ms = round(total_duration * 1000)
    total_gap_ms = expected_total_ms - combined_ms
    if total_gap_ms < 0:
        print(f"  Final: adjusting combined audio from {combined_ms}ms to {expected_total_ms}ms ({total_gap_ms:+d}ms)")
        combined = _rubberband_stretch(combined, combined_ms / expected_total_ms)

    combined.export(output_path, format="mp3")


def synthesize_google_timed(
    subtitles: list[Subtitle],
    total_duration: float,
    output_path: str,
    voice_name: str = "en-US-Studio-Q",
) -> None:
    """Two-pass SSML synthesis that matches the original video's speech pacing.

    When the full SSML exceeds 5000 bytes (Google's hard limit), subtitles are
    split into multiple chunks that are each within the limit.  Both passes
    operate on the same chunks; pass-2 audio is concatenated in order.

    Pass 1 — one API call per chunk, <break> tags only, no prosody.
             Total speech duration = sum(chunk durations) − total_silence.

    Pass 2 — same chunks, every text segment wrapped in <prosody rate="X%">:

        prosody_rate = speech_duration_pass1 / target_speech_duration
        target_speech_duration = total_duration − total_break_duration

    Inter-chunk gaps are preserved as the leading <break> of the next chunk,
    so the silence layout is identical across both passes.
    """
    from google.cloud import texttospeech
    from pydub import AudioSegment  # type: ignore

    project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
    client = texttospeech.TextToSpeechClient(
        client_options={"quota_project_id": project_id}
    )
    lang = "-".join(voice_name.split("-")[:2])
    voice_params = texttospeech.VoiceSelectionParams(language_code=lang, name=voice_name)
    audio_cfg = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    def call(ssml: str) -> bytes:
        return client.synthesize_speech(
            input=texttospeech.SynthesisInput(ssml=ssml),
            voice=voice_params,
            audio_config=audio_cfg,
        ).audio_content

    silences = _timed_silences(subtitles, total_duration)

    chunks = _ssml_chunks(subtitles, silences)
    if len(chunks) > 1:
        print(f"  SSML too large — split into {len(chunks)} chunks")

    # Pass 1: for each chunk measure actual audio duration, derive per-chunk prosody rate
    prosody_rates: list[float] = []
    total_dur1 = 0.0
    for chunk_subs, chunk_silences in chunks:
        audio = call(_timed_ssml(chunk_subs, chunk_silences, prosody_rate=None, ignore_silences=True))
        chunk_dur = len(AudioSegment.from_mp3(io.BytesIO(audio))) / 1000.0
        total_dur1 += chunk_dur

        chunk_target = (chunk_subs[-1].end - chunk_subs[0].start + chunk_silences[0] + chunk_silences[-1]) if len(chunk_subs) > 0 else 0.0
        rate = chunk_dur / chunk_target if chunk_target > 0 else 1.0
        prosody_rates.append(rate)

    print(f"  Pass 1: {total_dur1:.2f}s total  |  target {total_duration:.2f}s")
    rates_str = ", ".join(f"{r * 100:.1f}%" for r in prosody_rates)
    print(f"  Pass 2: per-chunk prosody rates [{rates_str}]")

    # Pass 2: apply per-chunk prosody rates and concatenate
    combined = AudioSegment.empty()
    for (chunk_subs, chunk_silences), rate in zip(chunks, prosody_rates):
        audio = call(_timed_ssml(chunk_subs, chunk_silences, prosody_rate=rate, ignore_silences=True))
        segment = AudioSegment.from_mp3(io.BytesIO(audio))

        expected_ms = ((chunk_subs[-1].end - chunk_subs[0].start + chunk_silences[0] + chunk_silences[-1]) if len(
            chunk_subs) > 0 else 0.0) * 1000
        actual_ms = len(segment)
        gap_ms = expected_ms - actual_ms
        if gap_ms > 0:
            print(f"  Pass 2 chunk: padding {gap_ms}ms to match expected {expected_ms}ms (got {actual_ms}ms)")
            segment += AudioSegment.silent(duration=gap_ms)
        elif gap_ms < 0:
            overshoot_ms = -gap_ms
            print(f"  Pass 2 chunk: compressing {overshoot_ms}ms overshoot to fit {expected_ms}ms (got {actual_ms}ms)")
            segment = _rubberband_stretch(segment, actual_ms / expected_ms)

        combined += segment

    combined_ms = len(combined)
    expected_total_ms = round(total_duration * 1000)
    total_gap_ms = expected_total_ms - combined_ms
    if total_gap_ms < 0:
        print(f"  Final: adjusting combined audio from {combined_ms}ms to {expected_total_ms}ms ({total_gap_ms:+d}ms)")
        combined = _rubberband_stretch(combined, combined_ms / expected_total_ms)

    combined.export(output_path, format="mp3")


def _rubberband_stretch(segment, time_ratio: float):
    """Return `segment` time-stretched by `time_ratio` with pitch preserved.

    time_ratio > 1 → shorter (compress); time_ratio < 1 → longer (expand).
    """
    import numpy as np
    import pyrubberband as pyrb
    from pydub import AudioSegment  # type: ignore

    sr = segment.frame_rate
    max_val = float(2 ** (8 * segment.sample_width - 1))
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32) / max_val
    if segment.channels == 2:
        samples = samples.reshape((-1, 2))
    stretched = pyrb.time_stretch(samples, sr, time_ratio)
    if segment.channels == 2:
        stretched = stretched.reshape(-1)
    raw = np.clip(stretched * max_val, -max_val, max_val - 1).astype(np.int16).tobytes()
    return AudioSegment(
        data=raw,
        sample_width=2,
        frame_rate=sr,
        channels=segment.channels,
    )


_CHUNK_SPAN_S = 10.0  # soft target: break chunks at this many seconds of span


def _ssml_chunks(
    subtitles: list[Subtitle],
    silences: list[float],
) -> list[tuple[list[Subtitle], list[float]]]:
    """Pack subtitles into chunks, preferring sentence boundaries or ~10-second
    spans, hard-capped at _SSML_MAX_BYTES (measured against pass-2 SSML).

    Each chunk's silence list has length len(chunk_subs) + 1:
      [before_sub0, gap_0→1, …, gap_(n-2)→(n-1), trailing]

    For non-final chunks the trailing silence is 0 — the gap to the next
    subtitle becomes the leading silence of the next chunk, so no silence
    is lost at chunk boundaries.
    """
    n = len(subtitles)
    chunks: list[tuple[list[Subtitle], list[float]]] = []
    start = 0

    while start < n:
        chunk_start_time = subtitles[start].start
        end = start

        while end < n:
            end += 1
            is_last = end == n
            chunk_subs = subtitles[start:end]
            chunk_silences = silences[start:end] + [silences[end] if is_last else 0.0]

            # Hard limit: size-check against pass-2 SSML (larger due to prosody tags)
            if len(_timed_ssml(chunk_subs, chunk_silences, prosody_rate=1.0).encode()) > _SSML_MAX_BYTES:
                end -= 1
                break

            # Soft break: sentence boundary
            last_text = subtitles[end - 1].text.strip()
            if last_text and last_text[-1] in ".!?":
                break

            # Soft break: span from chunk start exceeds target seconds
            if subtitles[end - 1].end - chunk_start_time >= _CHUNK_SPAN_S:
                break

        if end == start:
            end = start + 1  # safety: single subtitle always forms its own chunk

        is_last = end == n
        chunks.append((
            subtitles[start:end],
            silences[start:end] + [silences[end] if is_last else 0.0],
        ))
        start = end

    return chunks


# Google TTS accepts <break> values up to 10 s; stay conservative
_MAX_BREAK_S = 5.0


def _timed_silences(subtitles: list[Subtitle], total_duration: float) -> list[float]:
    """Return silence durations to insert: [before_sub0, gap_0→1, …, after_last_sub].
    Length is always len(subtitles) + 1."""
    result = [max(0.0, subtitles[0].start)]
    for a, b in zip(subtitles, subtitles[1:]):
        result.append(max(0.0, b.start - a.end))
    result.append(max(0.0, total_duration - subtitles[-1].end))
    return result


def _break_tag(seconds: float) -> str:
    """One or more <break> tags totalling `seconds`, split at _MAX_BREAK_S each."""
    if seconds <= 0:
        return ""
    tags, remaining = [], seconds
    while remaining > 0.001:
        chunk = min(remaining, _MAX_BREAK_S)
        tags.append(f'<break time="{chunk:.3f}s"/>')
        remaining -= chunk
    return "".join(tags)


def _timed_ssml(
    subtitles: list[Subtitle],
    silences: list[float],
    prosody_rate: float | None,
    ignore_silences: bool = False,
) -> str:
    """Build the full SSML string for a timed synthesis call.

    silences has len(subtitles) + 1 entries (before first, between each pair,
    after last).  When prosody_rate is set, each text segment is wrapped in
    <prosody rate="X%">; otherwise text is emitted as-is.
    """
    parts = ["<speak>"]
    for i, sub in enumerate(subtitles):
        if not ignore_silences:
            parts.append(_break_tag(silences[i]))
        text = sub.text
        if prosody_rate is not None:
            parts.append(f'<prosody rate="{prosody_rate * 100:.1f}%">{text}</prosody>')
        else:
            parts.append(text)
    if not ignore_silences:
        parts.append(_break_tag(silences[-1]))
    parts.append("</speak>")
    return "".join(parts)

