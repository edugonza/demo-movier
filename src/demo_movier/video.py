"""FFmpeg wrappers for audio extraction, subtitle burning, and audio replacement."""
from __future__ import annotations

import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg error:\n{result.stderr[-2000:]}"
        )


def extract_audio(video_path: str, output_wav: str) -> None:
    """Extract audio track as 16 kHz mono WAV (optimal for STT)."""
    _run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav,
    ])


def video_duration(video_path: str) -> float:
    """Return video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def burn_subtitles(
    video_path: str,
    srt_path: str,
    output_path: str,
    font_name: str = "Arial",
    font_size: int = 22,
    primary_color: str = "&H00FFFFFF",   # white
    outline_color: str = "&H00000000",   # black
    outline_width: int = 2,
    margin_v: int = 30,
    bold: bool = True,
) -> None:
    """Hardcode subtitles into the video (no separate SRT file needed by viewers).

    Color format is BGR with alpha prefix: &HAABBGGRR
      white = &H00FFFFFF   yellow = &H0000FFFF   cyan = &H00FFFF00
    Alignment=2 → bottom-center (standard subtitle position).
    """
    # Escape the SRT path for the subtitles filter (colons must be escaped on Windows)
    safe_srt = str(Path(srt_path).resolve()).replace("\\", "/").replace(":", "\\:")

    style = (
        f"FontName={font_name},"
        f"FontSize={font_size},"
        f"PrimaryColour={primary_color},"
        f"OutlineColour={outline_color},"
        f"Outline={outline_width},"
        f"Bold={'1' if bold else '0'},"
        f"MarginV={margin_v},"
        f"Alignment=2"
    )

    _run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"subtitles={safe_srt}:force_style='{style}'",
        "-c:a", "copy",
        output_path,
    ])


def replace_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
    keep_original: bool = False,
    original_volume: float = 0.0,   # 0.0 = silent, 0.15 = quiet background
) -> None:
    """Replace (or mix) the video audio track.

    keep_original=False  → pure synthetic voice (default for demo videos)
    keep_original=True   → mix synthetic voice with original at original_volume
    """
    if not keep_original or original_volume == 0.0:
        _run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-shortest",
            output_path,
        ])
    else:
        # Mix original + synthetic with volume control
        _run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-filter_complex",
            f"[0:a]volume={original_volume}[orig];[orig][1:a]amix=inputs=2:duration=first[aout]",
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            output_path,
        ])
