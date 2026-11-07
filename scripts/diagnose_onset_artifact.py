#!/usr/bin/env python
"""Ищет общее начало в готовых реконструкциях.

Результат относится только к проверенным файлам, а не ко всем выходам модели.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


def correlation_range(signals: list[np.ndarray], sr: int, begin: float, end: float) -> dict:
    slices = [x[int(begin * sr):int(end * sr)] for x in signals]
    values = [float(np.corrcoef(a, b)[0, 1]) for i, a in enumerate(slices)
              for b in slices[i + 1:] if np.std(a) > 1e-10 and np.std(b) > 1e-10]
    if not values:
        return {"pairs": 0, "median": None, "minimum": None, "maximum": None}
    return {"pairs": len(values), "median": float(np.median(values)),
            "minimum": min(values), "maximum": max(values)}


def analyze(directory: Path) -> dict:
    inputs = sorted(directory.glob("crop_*_input.wav"))
    if len(inputs) < 2:
        raise ValueError("At least two matching input/output pairs are needed")
    rows, outputs, sample_rate = [], [], None
    for path in inputs:
        out = path.with_name(path.name.replace("_input.wav", "_reconstruction.wav"))
        x, sr = sf.read(str(path))
        y, out_sr = sf.read(str(out))
        if x.ndim != 1 or y.ndim != 1 or sr != out_sr or len(x) != len(y) or len(x) < int(2.8 * sr):
            raise ValueError(f"Expected equal mono WAVs >= 2.8 s: {path}, {out}")
        if sample_rate is not None and sample_rate != sr:
            raise ValueError("Different sample rates")
        sample_rate = sr
        n = int(.05 * sr)
        rows.append({"input": path.name, "input_first_50ms_rms": rms(x[:n]),
                     "output_first_50ms_rms": rms(y[:n]),
                     "input_1500_2500ms_rms": rms(x[int(1.5 * sr):int(2.5 * sr)]),
                     "output_1500_2500ms_rms": rms(y[int(1.5 * sr):int(2.5 * sr)])})
        outputs.append(y)
    return {"sample_rate": sample_rate, "rows": rows,
            "output_cross_note_correlation_first_50ms": correlation_range(outputs, sample_rate, 0, .05),
            "output_cross_note_correlation_400_600ms": correlation_range(outputs, sample_rate, .4, .6),
            "output_cross_note_correlation_2500_2800ms": correlation_range(outputs, sample_rate, 2.5, 2.8)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.directory)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
