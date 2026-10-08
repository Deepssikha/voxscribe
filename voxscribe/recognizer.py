"""Speech recognition engine (Whisper through faster-whisper, beam search)."""
import os

import numpy as np
from faster_whisper import WhisperModel


class Recognizer:
    def __init__(self, model_size: str | None = None, device: str = "cpu",
                 compute_type: str = "int8"):
        self.model_size = model_size or os.environ.get("VOXSCRIBE_MODEL", "base.en")
        self.model = WhisperModel(self.model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, beam_size: int = 5) -> str:
        """Transcribe a mono 16 kHz float32 signal into text."""
        segments, _info = self.model.transcribe(
            audio.astype(np.float32),
            language="en",
            beam_size=beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
