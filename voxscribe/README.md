# VoxScribe: Real-Time Speech Recognition and Voice-Command System

Project by Deepsikha C D (RA2311003050046) and Chinmayi (RA2311003050359).

VoxScribe records speech in the browser, cleans the audio, recognises it with a
pre-trained Whisper model (beam search decoding) and shows live text. Spoken
phrases such as "save note" or "search for ..." run actions through fuzzy matching.

## Pipeline

1. `static/index.html` captures the microphone, cuts speech at pauses and sends 16 kHz WAV clips
2. `preprocessing.py` converts to mono 16 kHz, reduces noise, trims silence, normalises loudness
3. `features.py` computes 13 MFCCs plus deltas (25 ms frames, 10 ms shift)
4. `recognizer.py` decodes speech to text with faster-whisper
5. `commands.py` decides whether the text is a command or dictation
6. `app.py` is the Flask server that ties these together

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 in Chrome or Edge and allow the microphone.
The first request downloads the model (about 150 MB for `base.en`).

Choose a different model with an environment variable, for example
`VOXSCRIBE_MODEL=small.en python app.py` (larger = more accurate, slower).

## Voice commands

Start listening with the button or the M key. Then say:

| Phrase | Action |
|---|---|
| save note | Keep current text in the notes list |
| clear text | Empty the transcript |
| copy text | Copy the transcript |
| download text | Save a .txt file |
| open settings | Show or hide settings |
| search for ... | Web search for the words that follow |
| stop listening | Turn the microphone off |

## Evaluation

Record clips, list them in a CSV (see `data/manifest.example.csv`), then run:

```bash
python evaluate.py data/manifest.csv
```

It prints WER per condition (quiet, noisy, accents), mean processing time and
command accuracy. Paste these numbers into the Testing and Results section of the report.

## Tests

```bash
pytest tests
```

## Notes

* Whisper computes its own log-Mel features internally. The MFCC module is used for analysis
  and reporting, as described in the report.
* To fine-tune on your own recordings, use the Hugging Face `transformers` Whisper fine-tuning
  recipe with the same datasets listed in the report.
