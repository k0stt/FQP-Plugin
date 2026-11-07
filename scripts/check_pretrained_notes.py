#!/usr/bin/env python
"""Проверяет тишину и одиночные ноты на предобученной RAVE."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from diagnose_reconstruction import compare as compare_pitch
from rave_reconstruct import encode_deterministic, spectral_metrics


def rms_first_50ms(x: np.ndarray, sr: int) -> float:
    segment = x[:int(.05 * sr)]
    return float(np.sqrt(np.mean(segment * segment)))


def correlation(x: np.ndarray, y: np.ndarray, sr: int) -> float | None:
    a, b = x[:int(.05 * sr)], y[:int(.05 * sr)]
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def save(path: Path, x: np.ndarray, sr: int, gain: float) -> None:
    y = x.copy() * gain
    n = int(.01 * sr)
    envelope = np.sin(np.linspace(0, np.pi / 2, n)) ** 2
    y[:n] *= envelope
    y[-n:] *= envelope[::-1]
    sf.write(str(path), y, sr, subtype="FLOAT")


def reconstruct_fresh(model_path: Path, x: np.ndarray, seed: int = 111) -> tuple[np.ndarray, int]:
    """Загружает модель заново, чтобы состояние не переходило между файлами."""
    model = torch.jit.load(str(model_path), map_location="cpu").eval()
    sr = int(model.sr)
    with torch.inference_mode(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        z = encode_deterministic(model, torch.from_numpy(x.copy()).reshape(1, 1, -1))
        y = model.decode(z, True).squeeze().numpy().copy()
    del model
    if len(y) != len(x) or not np.isfinite(y).all():
        raise ValueError("Некорректный выход модели")
    return y, sr


def run(model_path: Path, input_dir: Path, output: Path) -> dict:
    files = sorted(input_dir.glob("*.wav"), key=lambda path: (not path.stem.startswith("piano_"), path.name))
    if not files:
        raise ValueError("Не найдены подготовленные WAV")
    torch.set_num_threads(2)
    first, sr = sf.read(str(files[0]), dtype="float32")
    if first.ndim != 1 or len(first) != 131072:
        raise ValueError(f"Неожиданный формат: {files[0]}")
    zero = np.zeros_like(first)
    zero_output, model_sr = reconstruct_fresh(model_path, zero)
    if model_sr != sr:
        raise ValueError("Частота WAV не совпадает с частотой модели")
    source, reconstructed = [], []
    for path in files:
        x, file_sr = sf.read(str(path), dtype="float32")
        if file_sr != sr or x.ndim != 1 or len(x) != len(zero):
            raise ValueError(f"Неожиданный формат: {path}")
        y, model_sr = reconstruct_fresh(model_path, x)
        if model_sr != sr:
            raise ValueError(f"Частота модели изменилась: {path}")
        source.append((path, x))
        reconstructed.append(y)
    zero_repeat, _ = reconstruct_fresh(model_path, zero)
    if not np.array_equal(zero_output, zero_repeat):
        raise ValueError("На двух свежих экземплярах выход на тишине различается")

    output.mkdir(parents=True, exist_ok=True)
    gain = min(1., .9 / max(float(np.max(np.abs(x))) for x in reconstructed + [zero_output]))
    rows = []
    for (path, x), y in zip(source, reconstructed):
        save(output / f"{path.stem}_input.wav", x, sr, gain)
        save(output / f"{path.stem}_reconstruction.wav", y, sr, gain)
        rows.append({"source": str(path), "first_50ms_input_rms": rms_first_50ms(x, sr),
                     "first_50ms_output_rms": rms_first_50ms(y, sr),
                     "first_50ms_max_difference_from_zero_output": float(np.max(
                         np.abs(y[:int(.05 * sr)] - zero_output[:int(.05 * sr)]))),
                     "first_50ms_correlation_with_zero_output": correlation(y, zero_output, sr),
                     "spectral": spectral_metrics(x, y, sr), "pitch_and_harmonic": compare_pitch(x, y, sr)})
    save(output / "zero_input.wav", zero, sr, gain)
    save(output / "zero_input_reconstruction.wav", zero_output, sr, gain)
    correlations = [correlation(reconstructed[i], reconstructed[j], sr)
                    for i in range(len(reconstructed)) for j in range(i + 1, len(reconstructed))]
    valid = [c for c in correlations if c is not None]
    result = {"model": str(model_path), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "protocol": "fresh TorchScript instance per input; posterior mean; seed 111; decode(from_forward=True)",
              "zero_first_equals_zero_after_notes": True, "sample_rate": sr, "common_listening_gain": gain,
              "zero_input_output_first_50ms_rms": rms_first_50ms(zero_output, sr),
              "zero_input_output_tail_1500_2500ms_rms": float(np.sqrt(np.mean(
                  zero_output[int(1.5 * sr):int(2.5 * sr)] ** 2))),
              "median_pairwise_first_50ms_correlation": float(np.median(valid)) if valid else None,
              "samples": rows}
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/musicnet.ts"))
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.model, args.input_dir, args.output), indent=2))
