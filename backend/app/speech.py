"""Local speech-to-text (Whisper base.en via transformers). Audio never leaves the machine."""
from __future__ import annotations

import io
import threading
import wave

import numpy as np

from . import config

WHISPER_ID = "openai/whisper-base.en"
_lock = threading.Lock()
_asr = None


def _load():
    global _asr
    with _lock:
        if _asr is None:
            from transformers import pipeline
            _asr = pipeline("automatic-speech-recognition", model=WHISPER_ID, device=config.DEVICE)
    return _asr


def transcribe_wav(data: bytes) -> str:
    with wave.open(io.BytesIO(data)) as w:
        if w.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM WAV")
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1)
    audio = pcm.astype(np.float32) / 32768.0
    if sr != 16000:                       # simple linear resample
        n = int(len(audio) * 16000 / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    if len(audio) < 1600 or np.abs(audio).max() < 0.01:
        return ""
    out = _load()({"raw": audio, "sampling_rate": 16000})
    return out["text"].strip()
