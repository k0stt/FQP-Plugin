#!/usr/bin/env python
"""Проверка декодера: zero_input.wav — тишина, zero_input_reconstruction.wav — выход модели."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import gin
import numpy as np
import soundfile as sf
import torch
from rave_reconstruct import spectral_metrics


def rms(x: np.ndarray, sr: int, start: float, end: float) -> float:
    a = x[int(start * sr):int(end * sr)]
    return float(np.sqrt(np.mean(a * a)))


def compare(x: np.ndarray, y: np.ndarray, sr: int) -> dict:
    return {"first_50ms_correlation": float(np.corrcoef(x[:int(.05 * sr)], y[:int(.05 * sr)])[0, 1]),
            "tail_2500_2800ms_correlation": float(np.corrcoef(
                x[int(2.5 * sr):int(2.8 * sr)], y[int(2.5 * sr):int(2.8 * sr)])[0, 1])}


def build_signals(x, other, full, second, silence, silence_noise_off, alt_seed, no_noise, contextual):
    return {"input": x, "other_input": other, "full": full, "other_full": second,
            "zero_input": np.zeros_like(x), "zero_input_reconstruction": silence,
            "zero_input_noise_off": silence_noise_off,
            "seed_777": alt_seed, "noise_off": no_noise, "context": contextual,
            "zero_input_output_subtracted": full - silence,
            "other_zero_input_output_subtracted": second - silence}


def run(checkpoint: Path, config: Path, note: Path, other_note: Path, output: Path) -> dict:
    import rave

    torch.set_num_threads(2)
    gin.parse_config_file(str(config))
    model = rave.RAVE(n_channels=1)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    x, sr = sf.read(str(note), dtype="float32")
    other, other_sr = sf.read(str(other_note), dtype="float32")
    if sr != model.sr or other_sr != sr or x.ndim != 1 or other.ndim != 1 or len(x) != 131072 or len(other) != len(x):
        raise ValueError("Нужны две моно-ноты по 131072 отсчётов с частотой модели")

    def decode(signal: np.ndarray, seed: int, noise_off: bool = False) -> np.ndarray:
        z = model.encode(torch.from_numpy(signal.copy()).reshape(1, 1, -1))
        mean = z[:, :z.shape[1] // 2]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if noise_off:
                noise = model.decoder.noise_module
                original = noise.forward
                with patch.object(noise, "forward", side_effect=lambda features: torch.zeros_like(original(features))):
                    out = model.decode(mean)
            else:
                out = model.decode(mean)
        return out.squeeze().numpy().copy()

    pad = 43 * 512  # ~0.499 с с обеих сторон; длина остаётся кратной латентному шагу
    with torch.inference_mode():
        full = decode(x, 111)
        second = decode(other, 111)
        silence = decode(np.zeros_like(x), 111)
        silence_noise_off = decode(np.zeros_like(x), 111, noise_off=True)
        alt_seed = decode(x, 777)
        no_noise = decode(x, 111, noise_off=True)
        contextual = decode(np.pad(x, (pad, pad)), 111)[pad:pad + len(x)]
    signals = build_signals(x, other, full, second, silence, silence_noise_off,
                            alt_seed, no_noise, contextual)
    if any(not np.isfinite(a).all() or len(a) != len(x) for a in signals.values()):
        raise ValueError("Неверная длина или неконечный сигнал")
    output.mkdir(parents=True, exist_ok=True)
    gain = min(1., .9 / max(float(np.max(np.abs(a))) for a in signals.values()))
    fade = np.sin(np.linspace(0, np.pi / 2, int(.01 * sr))) ** 2
    for name, signal in signals.items():
        audio = signal.copy() * gain
        audio[:len(fade)] *= fade
        audio[-len(fade):] *= fade[::-1]
        sf.write(str(output / f"{name}.wav"), audio, sr, subtype="FLOAT")
    report = {"checkpoint": str(checkpoint), "step": int(state["global_step"]), "sample_rate": sr,
              "common_listening_gain": gain, "context_extra_each_side_samples": pad,
              "signals": {name: {"first_50ms_rms": rms(a, sr, 0, .05),
                                 "note_400_600ms_rms": rms(a, sr, .4, .6),
                                 "tail_1500_2500ms_rms": rms(a, sr, 1.5, 2.5)}
                          for name, a in signals.items()},
              "pairs": {name: compare(full, signals[name], sr)
                        for name in ("other_full", "zero_input_reconstruction", "zero_input_noise_off",
                                     "seed_777", "noise_off", "context")},
              "subtraction_control": {
                  "first": {"raw": spectral_metrics(x, full, sr),
                            "silence_subtracted": spectral_metrics(x, full - silence, sr)},
                  "second": {"raw": spectral_metrics(other, second, sr),
                             "silence_subtracted": spectral_metrics(other, second - silence, sr)}}}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--note", type=Path, required=True)
    parser.add_argument("--other-note", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.config, args.note, args.other_note, args.output), indent=2))
