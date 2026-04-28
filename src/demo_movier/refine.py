"""Subtitle refinement: filler removal and sentence-level joining.

Two backends:
  rules — offline, fast. Strips clear hesitation sounds (um, uh, hmm…)
          and joins segments that end mid-sentence.
  llm   — uses Gemini Flash via Vertex AI, giving cleaner and more
          natural results. Reuses GOOGLE_CLOUD_PROJECT + ADC — no extra
          API key required.
          Install: uv sync --extra llm
"""
from __future__ import annotations

import json
import re

from demo_movier.subtitles import Subtitle


# ──────────────────────────────────────────────────────────────────────────────
# Rules backend
# ──────────────────────────────────────────────────────────────────────────────

# Unconditional: clear hesitation sounds
_HESITATIONS = re.compile(r'\b(u+m+|u+h+|h+m+|a+h+|e+r+m?|m+hm+)\b[,]?', re.IGNORECASE)

# Sentence-boundary fillers: only strip when they appear at the very start or end
_START_FILLERS = re.compile(
    r'^(so[,\s]+|okay[,\s]+|alright[,\s]+|right[,\s]+|now[,\s]+|well[,\s]+)',
    re.IGNORECASE,
)
_END_FILLERS = re.compile(
    r'([,\s]+(right|okay|you know|you see))[.!?]?$',
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    text = _HESITATIONS.sub('', text)
    text = _START_FILLERS.sub('', text)
    text = _END_FILLERS.sub('', text)
    text = re.sub(r'\s{2,}', ' ', text)          # collapse spaces
    text = re.sub(r'\s+([,.!?])', r'\1', text)   # fix space before punctuation
    text = re.sub(r'^[,\s]+', '', text)           # leading comma/space
    return text.strip()


def _is_sentence_end(text: str) -> bool:
    return bool(re.search(r'[.!?]\s*$', text))


def refine_rules(
    subs: list[Subtitle],
    join_pause_threshold: float = 0.8,
) -> list[Subtitle]:
    """Remove hesitation sounds and join mid-sentence segments.

    Two passes:
    1. Clean each subtitle's text; drop any that become empty.
    2. Greedily join consecutive subtitles while the current segment does
       not end a sentence AND the gap to the next is < join_pause_threshold.
       The joined segment inherits the first sub's start and the last sub's end.
    """
    # Pass 1: clean
    cleaned = [Subtitle(s.index, s.start, s.end, _clean(s.text)) for s in subs]
    cleaned = [s for s in cleaned if s.text]

    # Pass 2: join
    merged: list[Subtitle] = []
    i = 0
    while i < len(cleaned):
        cur = cleaned[i]
        while (
            i + 1 < len(cleaned)
            and not _is_sentence_end(cur.text)
            and cleaned[i + 1].start - cur.end < join_pause_threshold
        ):
            i += 1
            nxt = cleaned[i]
            text = cur.text.rstrip(' ,') + ' ' + nxt.text
            cur = Subtitle(cur.index, cur.start, nxt.end, text)
        merged.append(cur)
        i += 1

    _renumber(merged)
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# LLM backend (Gemini Flash via Vertex AI)
# ──────────────────────────────────────────────────────────────────────────────

_BATCH_SIZE = 40  # subtitles per API call

_SYSTEM_PROMPT = """\
You are a subtitle editor cleaning auto-generated speech-to-text subtitles for a tech demo video.

Tasks — apply ALL of them to every subtitle in the batch:
1. Remove hesitation sounds: um, uh, hmm, hm, ah, er, erm, mhm and similar.
2. Remove sentence-boundary filler phrases: "so,", "okay,", "right?",
   "you know", "I mean" (when used as filler at the start or end of a line).
3. Join consecutive subtitle entries that belong to the same sentence.
   A segment that ends mid-sentence (no . ! ?) should be merged with the
   next one. Use the FIRST segment's start time and the LAST segment's end time.
4. Do NOT paraphrase, summarise, or change meaning.
   Only remove fillers and join fragments. Preserve all other words.

Return a JSON array where each element is:
  {"id": <int>, "start": <float>, "end": <float>, "text": "<string>"}

Omit entries whose text becomes empty after cleaning."""


def refine_llm(
    subs: list[Subtitle],
    model: str = "gemini-2.0-flash",
) -> list[Subtitle]:
    """Clean and join subtitles using Gemini Flash (Vertex AI).

    Reuses GOOGLE_CLOUD_PROJECT and Application Default Credentials — no
    separate API key needed.  Batches subtitles to stay within the model's
    context window and requests JSON output directly.
    """
    import os
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore

    client = genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location="us-central1",
    )

    batches = _make_batches(subs, _BATCH_SIZE)
    result: list[Subtitle] = []

    for i, batch in enumerate(batches):
        print(f"  LLM refine: batch {i + 1}/{len(batches)} ({len(batch)} segments) …")
        payload = [
            {"id": s.index, "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
            for s in batch
        ]
        response = client.models.generate_content(
            model=model,
            contents=json.dumps(payload, ensure_ascii=False),
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
            ),
        )
        items = json.loads(response.text)
        result.extend(
            Subtitle(index=it["id"], start=it["start"], end=it["end"], text=it["text"])
            for it in items
        )

    _renumber(result)
    return result


def _make_batches(subs: list[Subtitle], size: int) -> list[list[Subtitle]]:
    """Split into batches of `size`, trying to break at sentence boundaries."""
    batches: list[list[Subtitle]] = []
    start = 0
    n = len(subs)
    while start < n:
        end = min(start + size, n)
        if end < n:
            for k in range(end - 1, start, -1):
                if _is_sentence_end(subs[k].text):
                    end = k + 1
                    break
        batches.append(subs[start:end])
        start = end
    return batches


def _renumber(subs: list[Subtitle]) -> None:
    for i, s in enumerate(subs, 1):
        s.index = i
