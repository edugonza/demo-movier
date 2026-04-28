"""Subtitle grouping and SRT generation."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Word:
    text: str
    start: float  # seconds
    end: float    # seconds


@dataclass
class Subtitle:
    index: int
    start: float
    end: float
    text: str


def group_into_subtitles(
    words: list[Word],
    max_words: int = 8,
    max_chars: int = 60,
    pause_threshold: float = 0.6,  # new subtitle on silence gap > this
) -> list[Subtitle]:
    """Group word timestamps into subtitle blocks."""
    if not words:
        return []

    subtitles: list[Subtitle] = []
    group: list[Word] = [words[0]]

    for word in words[1:]:
        gap = word.start - group[-1].end
        current_text = " ".join(w.text for w in group) + " " + word.text
        split = (
            len(group) >= max_words
            or len(current_text) > max_chars
            or gap > pause_threshold
        )
        if split:
            subtitles.append(_make(len(subtitles) + 1, group))
            group = [word]
        else:
            group.append(word)

    if group:
        subtitles.append(_make(len(subtitles) + 1, group))

    return subtitles


def _make(index: int, words: list[Word]) -> Subtitle:
    return Subtitle(
        index=index,
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(w.text for w in words),
    )


def to_srt(subtitles: list[Subtitle]) -> str:
    blocks = []
    for sub in subtitles:
        blocks.append(
            f"{sub.index}\n"
            f"{_tc(sub.start)} --> {_tc(sub.end)}\n"
            f"{sub.text}\n"
        )
    return "\n".join(blocks)


def load_srt(path: str) -> list[Subtitle]:
    """Parse an existing SRT file into Subtitle objects."""
    text = open(path, encoding="utf-8").read().strip()
    subtitles: list[Subtitle] = []
    for block in text.split("\n\n"):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        index = int(lines[0].strip())
        start_s, end_s = lines[1].split(" --> ")
        content = " ".join(lines[2:]).strip()
        subtitles.append(Subtitle(
            index=index,
            start=_parse_tc(start_s.strip()),
            end=_parse_tc(end_s.strip()),
            text=content,
        ))
    return subtitles


def _tc(seconds: float) -> str:
    """Seconds → SRT timecode  HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _parse_tc(tc: str) -> float:
    """SRT timecode → seconds"""
    tc = tc.replace(",", ".")
    parts = tc.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
