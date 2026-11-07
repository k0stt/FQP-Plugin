#!/usr/bin/env python
"""Pitch-class diagnostics on listening pairs; not a perceptual quality score.

Run: .venv/Scripts/python scripts/diagnose_reconstruction.py outputs/stage1b/external_listen
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import stft


def pitch_classes(x: np.ndarray, sr: int) -> np.ndarray:
    """Average energy in 12 equal-tempered pitch classes, 80..2000 Hz."""
    f, _, z = stft(x, fs=sr, nperseg=8192, noverlap=6144, boundary=None)
    mask = (f >= 80) & (f <= 2000)
    midi = np.rint(69 + 12 * np.log2(f[mask] / 440)).astype(int)
    chroma = np.bincount(midi % 12, weights=(np.abs(z[mask]) ** 2).mean(axis=1), minlength=12)
    return chroma / (np.linalg.norm(chroma) + 1e-12)


def harmonic_energy_fraction(x: np.ndarray, sr: int) -> float:
    """Approximate median-filter HPSS harmonic fraction; not a quality score."""
    _, _, z = stft(x, fs=sr, nperseg=2048, noverlap=1536)
    spectrum = np.abs(z)
    harmonic = median_filter(spectrum, size=(1, 31))
    percussive = median_filter(spectrum, size=(31, 1))
    mask = harmonic ** 2 / (harmonic ** 2 + percussive ** 2 + 1e-12)
    return float(np.sum((spectrum * mask) ** 2) / (np.sum(spectrum ** 2) + 1e-12))


def compare(x: np.ndarray, y: np.ndarray, sr: int) -> dict:
    a, b = pitch_classes(x, sr), pitch_classes(y, sr)
    shifts = [float(np.dot(np.roll(a, i), b)) for i in range(12)]
    shift = int(np.argmax(shifts))
    return {"same_pitch_class_similarity": shifts[0],
            "best_shift_semitones": shift if shift <= 6 else shift - 12,
            "best_shift_similarity": shifts[shift],
            "input_dominant_pitch_class": int(np.argmax(a)),
            "output_dominant_pitch_class": int(np.argmax(b)),
            "input_harmonic_fraction": harmonic_energy_fraction(x, sr),
            "output_harmonic_fraction": harmonic_energy_fraction(y, sr),
            "input_peak": float(np.max(np.abs(x))), "output_peak": float(np.max(np.abs(y)))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    results = []
    for inp in sorted(args.directory.glob("crop_*_input.wav")):
        out = inp.with_name(inp.name.replace("_input.wav", "_reconstruction.wav"))
        x, sr = sf.read(str(inp))
        y, out_sr = sf.read(str(out))
        if sr != out_sr or x.ndim != 1 or y.ndim != 1:
            raise ValueError(f"Expected matching mono WAV files: {inp}, {out}")
        results.append({"file": inp.name, **compare(x, y, sr)})
    print(json.dumps(results, indent=2))
