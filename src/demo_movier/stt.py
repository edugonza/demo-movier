"""Speech-to-Text backends.

Supported:
  google  — Google Cloud Speech-to-Text v2, model Chirp 2.
            Requires: GOOGLE_CLOUD_PROJECT env var + ADC credentials.
            Chirp 2 gives word-level timestamps and is available in
            us-central1 / europe-west4.
  whisper — OpenAI Whisper running locally (no API key).
            Install extras: uv sync --extra whisper
"""
from __future__ import annotations

import os
from pathlib import Path

from demo_movier.subtitles import Word


# ---------------------------------------------------------------------------
# Google Cloud Speech-to-Text v2  (Chirp 2)
# ---------------------------------------------------------------------------

GOOGLE_CHUNK_SECONDS = 55  # stay safely under the 60s synchronous limit


def transcribe_google(
    audio_path: str,
    language: str = "en-US",
    project_id: str | None = None,
    location: str = "us-central1",
) -> list[Word]:
    """Transcribe with Google Chirp 2.  Audio must be WAV/FLAC mono 16 kHz.

    Automatically chunks audio longer than ~55 s so it fits within the
    synchronous recognition limit.  For very long files (> 10 min) consider
    using transcribe_google_gcs() instead.
    """
    from google.cloud import speech_v2
    from google.cloud.speech_v2.types import cloud_speech
    import wave, struct

    project_id = project_id or os.environ["GOOGLE_CLOUD_PROJECT"]
    client = speech_v2.SpeechClient(
        client_options={"api_endpoint": f"{location}-speech.googleapis.com", "quota_project_id": project_id}
    )

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=[language],
        model="chirp_2",
        features=cloud_speech.RecognitionFeatures(
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
        ),
    )
    recognizer = f"projects/{project_id}/locations/{location}/recognizers/_"

    chunks = _split_wav(audio_path, GOOGLE_CHUNK_SECONDS)
    all_words: list[Word] = []

    for i, (chunk_bytes, offset_seconds) in enumerate(chunks):
        if len(chunks) > 1:
            print(f"  chunk {i + 1}/{len(chunks)} (offset {offset_seconds:.1f}s) …")
        request = cloud_speech.RecognizeRequest(
            recognizer=recognizer,
            config=config,
            content=chunk_bytes,
        )
        response = client.recognize(request=request)
        for word in _google_words(response):
            all_words.append(Word(
                text=word.text,
                start=word.start + offset_seconds,
                end=word.end + offset_seconds,
            ))

    return all_words


def _split_wav(audio_path: str, chunk_seconds: int) -> list[tuple[bytes, float]]:
    """Split a WAV file into chunks of at most chunk_seconds.
    Returns list of (wav_bytes, start_offset_seconds)."""
    import wave, io

    with wave.open(audio_path, "rb") as wf:
        framerate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        frames_per_chunk = framerate * chunk_seconds

        chunks: list[tuple[bytes, float]] = []
        frame_pos = 0
        while frame_pos < n_frames:
            wf.setpos(frame_pos)
            raw = wf.readframes(frames_per_chunk)
            offset_sec = frame_pos / framerate

            buf = io.BytesIO()
            with wave.open(buf, "wb") as out:
                out.setnchannels(n_channels)
                out.setsampwidth(sampwidth)
                out.setframerate(framerate)
                out.writeframes(raw)
            chunks.append((buf.getvalue(), offset_sec))
            frame_pos += frames_per_chunk

    return chunks


def transcribe_google_gcs(
    gcs_uri: str,
    language: str = "en-US",
    project_id: str | None = None,
    location: str = "us-central1",
) -> list[Word]:
    """Long-form transcription via GCS (videos longer than ~60 s).

    Upload your audio to a GCS bucket first:
        gsutil cp audio.wav gs://your-bucket/audio.wav
    Then pass gcs_uri='gs://your-bucket/audio.wav'.
    """
    from google.cloud import speech_v2
    from google.cloud.speech_v2.types import cloud_speech

    project_id = project_id or os.environ["GOOGLE_CLOUD_PROJECT"]
    client = speech_v2.SpeechClient(
        client_options={"api_endpoint": f"{location}-speech.googleapis.com", "quota_project_id": project_id}
    )

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=[language],
        model="chirp_2",
        features=cloud_speech.RecognitionFeatures(
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
        ),
    )

    file_metadata = cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)

    request = cloud_speech.BatchRecognizeRequest(
        recognizer=f"projects/{project_id}/locations/{location}/recognizers/_",
        config=config,
        files=[file_metadata],
        recognition_output_config=cloud_speech.RecognitionOutputConfig(
            inline_response_config=cloud_speech.InlineOutputConfig(),
        ),
    )

    operation = client.batch_recognize(request=request)
    print("  Waiting for Google batch transcription…")
    response = operation.result(timeout=600)

    words: list[Word] = []
    for file_result in response.results.values():
        for result in file_result.transcript.results:
            words.extend(_google_words_from_result(result))
    return words


def _google_words(response) -> list[Word]:
    words: list[Word] = []
    for result in response.results:
        words.extend(_google_words_from_result(result))
    return words


def _google_words_from_result(result) -> list[Word]:
    words: list[Word] = []
    if not result.alternatives:
        return words
    for w in result.alternatives[0].words:
        words.append(Word(
            text=w.word,
            start=w.start_offset.total_seconds(),
            end=w.end_offset.total_seconds(),
        ))
    return words


# ---------------------------------------------------------------------------
# OpenAI Whisper  (local, no API key)
# ---------------------------------------------------------------------------

def transcribe_whisper(
    audio_path: str,
    model_size: str = "large-v3",
    language: str | None = None,
) -> list[Word]:
    """Transcribe locally with OpenAI Whisper.

    model_size options: tiny, base, small, medium, large-v3
    First run downloads the model weights (~1.5 GB for large-v3).
    """
    import whisper  # type: ignore

    model = whisper.load_model(model_size)
    kwargs: dict = {"word_timestamps": True}
    if language:
        kwargs["language"] = language
    result = model.transcribe(audio_path, **kwargs)

    words: list[Word] = []
    for segment in result["segments"]:
        for w in segment.get("words", []):
            words.append(Word(
                text=w["word"].strip(),
                start=w["start"],
                end=w["end"],
            ))
    return words
