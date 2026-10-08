"""Feature extraction: MFCCs with delta and delta-delta coefficients.

Frames are 25 ms long with a 10 ms shift, as described in the project report.
The Whisper model computes its own log-Mel spectrogram internally; these
MFCCs are used for analysis, visualisation and the /api/transcribe summary.
"""
import librosa
import numpy as np


def extract_mfcc(audio: np.ndarray, sr: int = 16000, n_mfcc: int = 13,
                 with_deltas: bool = True) -> np.ndarray:
    """Return an array of shape (n_features, n_frames)."""
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_mels=40,
        n_fft=int(0.025 * sr), hop_length=int(0.010 * sr),
    )
    if not with_deltas or mfcc.shape[1] < 9:
        return mfcc
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2])
