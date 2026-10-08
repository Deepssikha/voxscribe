"""VoxScribe - real-time speech recognition and voice-command web app."""
import io
import time

import numpy as np
import soundfile as sf
from flask import Flask, jsonify, request

from commands import match_command
from features import extract_mfcc
from preprocessing import TARGET_SR, preprocess, to_mono_16k
from recognizer import Recognizer

app = Flask(__name__, static_folder="static", static_url_path="")
_recognizer: Recognizer | None = None


def get_recognizer() -> Recognizer:
    """Load the model on first use so the server starts quickly."""
    global _recognizer
    if _recognizer is None:
        _recognizer = Recognizer()
    return _recognizer


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok", model=get_recognizer().model_size)


@app.post("/api/transcribe")
def transcribe():
    raw = request.get_data()
    if not raw:
        return jsonify(error="No audio received."), 400
    denoise = request.args.get("denoise", "1") == "1"
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception:
        return jsonify(error="Audio must be a valid WAV file."), 400

    started = time.perf_counter()
    audio = preprocess(to_mono_16k(audio, sr), TARGET_SR, denoise)
    if len(audio) < TARGET_SR * 0.3:
        return jsonify(text="", command=None, latency_ms=0, duration=0)

    mfcc = extract_mfcc(audio, TARGET_SR)
    text = get_recognizer().transcribe(audio)
    command = match_command(text)
    return jsonify(
        text=text,
        command=command,
        duration=round(len(audio) / TARGET_SR, 2),
        mfcc_shape=list(np.shape(mfcc)),
        latency_ms=round((time.perf_counter() - started) * 1000),
    )


if __name__ == "__main__":
    # Microphone access works on http://localhost without HTTPS.
    app.run(host="127.0.0.1", port=5000, debug=False)
