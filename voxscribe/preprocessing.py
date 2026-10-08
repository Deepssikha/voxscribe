"""Audio pre-processing: mono/16 kHz conversion, silence trimming,
loudness normalisation and spectral noise reduction."""
import librosa
import noisereduce as nr
import numpy as np

TARGET_SR = 16000


def to_mono_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return a mono float32 signal sampled at 16 kHz."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio.astype(np.float32)


def trim_silence(audio: np.ndarray, top_db: int = 35) -> np.ndarray:
    """Remove leading and trailing silence (keeps the clip if all quiet)."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed if len(trimmed) else audio


def normalise(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Scale the signal so its loudest sample reaches target_peak."""
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    return audio * (target_peak / peak) if peak > 1e-6 else audio


def reduce_noise(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Stationary spectral-gating noise reduction (fans, hum, room noise)."""
    return nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.8)


def preprocess(audio: np.ndarray, sr: int = TARGET_SR, denoise: bool = True) -> np.ndarray:
    """Full pre-processing chain used before recognition."""
    if denoise and len(audio) > sr // 4:
        audio = reduce_noise(audio, sr)
    audio = trim_silence(audio)
    return normalise(audio).astype(np.float32)
