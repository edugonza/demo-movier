# demo-movier

Pipeline to turn an already-edited demo video into one with **hardcoded subtitles** and optionally a **synthetic voice** replacing the original audio.

```
input.mp4 → transcribe → subtitles → burn → TTS → replace audio → final.mp4
```

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- FFmpeg (`brew install ffmpeg-full`)
- A Google Cloud project with **Speech-to-Text** and **Text-to-Speech** APIs enabled

```bash
gcloud services enable speech.googleapis.com texttospeech.googleapis.com \
  --project=YOUR_PROJECT_ID
```

## Setup

```bash
uv sync
cp .env.example .env
# fill in GOOGLE_CLOUD_PROJECT in .env
```

Authenticate with Google Cloud:

```bash
gcloud auth application-default login
```

## Usage

### Full pipeline in one command

```bash
uv run movier run demo.mp4
```

This produces `demo.final.mp4` with hardcoded subtitles and a synthetic voice. Add `--keep-intermediates` to also save the `.words.json`, `.srt`, and `.tts.mp3` files.

```bash
uv run movier run demo.mp4 \
  --voice en-US-Chirp3-HD-Aoede \   # newest Google voice
  --timed \                          # sync TTS to original subtitle timing
  --color yellow \
  --font-size 24 \
  --keep-intermediates
```

### Step by step

**1. Transcribe** — extract audio and get word-level timestamps:

```bash
uv run movier transcribe demo.mp4
# → demo.words.json
```

**2. Generate subtitles** — group words into an SRT file:

```bash
uv run movier subtitles demo.words.json
# or directly from a video:
uv run movier subtitles demo.mp4
# → demo.srt
```

**3. Burn subtitles** — hardcode into the video:

```bash
uv run movier burn demo.mp4 demo.srt
# → demo.subtitled.mp4
```

**4. Synthesise voice** — generate TTS audio from the SRT:

```bash
uv run movier voice demo.srt --voice en-US-Studio-Q
# with timed mode (each subtitle placed at its original timestamp):
uv run movier voice demo.srt --timed --video demo.mp4
# → demo.tts.mp3
```

**5. Replace audio**:

```bash
uv run movier replace demo.subtitled.mp4 demo.tts.mp3
# → demo.subtitled.revoiced.mp4

# or mix synthetic voice with original audio at low volume:
uv run movier replace demo.subtitled.mp4 demo.tts.mp3 --mix --original-volume 0.1
```

## STT backends

| Backend | Quality | Cost | Notes |
|---|---|---|---|
| `google` *(default)* | Excellent | ~$0.004/min | Chirp 2 model, auto-chunked for long videos |
| `whisper` | Excellent | Free | Runs locally; install with `uv sync --extra whisper` |

Switch with `--stt whisper` on any command that triggers transcription.

For videos longer than ~10 minutes, upload the audio to GCS first and use the `--gcs-uri` flag:

```bash
gsutil cp audio.wav gs://your-bucket/audio.wav
uv run movier transcribe demo.mp4 --gcs-uri gs://your-bucket/audio.wav
```

## TTS backends

| Backend | Voice quality | Cost | Notes |
|---|---|---|---|
| `google` *(default)* | Very good | ~$0.016/1k chars | Studio & Chirp3 HD voices |
| `elevenlabs` | Best | ~$0.18/1k chars | Install with `uv sync --extra elevenlabs` |

### Recommended Google voices

| Name | Voice ID | Style |
|---|---|---|
| Studio male | `en-US-Studio-Q` | Clear, neutral |
| Studio female | `en-US-Studio-O` | Clear, neutral |
| Chirp3 HD Aoede | `en-US-Chirp3-HD-Aoede` | Warm female, most natural |
| Chirp3 HD Charon | `en-US-Chirp3-HD-Charon` | Deep male |
| Chirp3 HD Fenrir | `en-US-Chirp3-HD-Fenrir` | Authoritative male |
| Chirp3 HD Kore | `en-US-Chirp3-HD-Kore` | Clear female |

### TTS timing modes

- **full** *(default)* — entire transcript synthesised as one audio file; pacing may differ from the original video
- **timed** (`--timed`) — each subtitle segment is synthesised separately and placed at its original timestamp; preserves the original pacing layout

## Subtitle customisation

| Option | Default | Notes |
|---|---|---|
| `--max-words` | 8 | Max words per subtitle block |
| `--max-chars` | 60 | Max characters per line |
| `--pause` | 0.6s | Start a new block when silence gap exceeds this |
| `--font-size` | 22 | |
| `--color` | `white` | `white`, `yellow`, or `cyan` |
| `--no-bold` | — | Bold is on by default |
| `--margin-v` | 30px | Distance from the bottom edge |

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | Yes (Google STT/TTS) | GCP project ID |
| `GOOGLE_APPLICATION_CREDENTIALS` | No | Path to service account JSON; not needed if using `gcloud auth application-default login` |
| `ELEVENLABS_API_KEY` | Only for ElevenLabs | API key from elevenlabs.io |
