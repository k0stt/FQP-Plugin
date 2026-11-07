#!/usr/bin/env python
"""Reconstruct with/without NoiseGeneratorV2, WITHOUT changing saved weights.

Diagnostic ablation only: a decoder trained WITH noise is not guaranteed to
sound right when its noise branch is silenced. Never use these values as proof
that a different architecture has been trained.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import gin
import numpy as np
import soundfile as sf
import soxr
import torch

from diagnose_reconstruction import compare
from rave_reconstruct import integrity, spectral_metrics


def disable_noise_forward(module):
    """Return a temporary mock leaving the module and every parameter intact."""
    original = module.forward
    return patch.object(module, "forward", side_effect=lambda x: torch.zeros_like(original(x)))


def centered_crop(path: Path, sr: int, n: int = 131072) -> np.ndarray:
    info = sf.info(str(path))
    start = max(0, (info.frames - int(3 * info.samplerate)) // 2)
    x, source_sr = sf.read(str(path), start=start, frames=int(3 * info.samplerate),
                           dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if source_sr != sr:
        x = soxr.resample(x, source_sr, sr)
    if len(x) < n:
        raise ValueError(f"Too short: {path}")
    return x[(len(x) - n) // 2: (len(x) - n) // 2 + n]


def save_triplet(directory: Path, index: int, arrays: list, sr: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    gain = min(1., .9 / max(max(float(np.max(np.abs(a))) for a in arrays), 1e-12))
    n = int(.010 * sr)
    env = np.sin(np.linspace(0, np.pi / 2, n)) ** 2
    for name, audio in zip(("input", "full", "noise_off"), arrays):
        y = audio.copy() * gain
        y[:n] *= env
        y[-n:] *= env[::-1]
        sf.write(str(directory / f"crop_{index:02d}_{name}.wav"), y, sr, subtype="FLOAT")


def run(checkpoint: Path, config: Path, files: list[Path], out: Path) -> dict:
    import rave

    gin.parse_config_file(str(config))
    model = rave.RAVE(n_channels=1)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    noise_module = model.decoder.noise_module
    if noise_module is None:
        raise ValueError("Checkpoint has no NoiseGeneratorV2 to ablate")
    results = []
    with torch.inference_mode():
        for index, path in enumerate(files):
            x = centered_crop(path, model.sr)
            z = model.encode(torch.from_numpy(x.copy()).reshape(1, 1, -1))
            mean = z[:, :z.shape[1] // 2]
            seed = 111 + index
            torch.manual_seed(seed)
            full = model.decode(mean).squeeze().numpy()
            torch.manual_seed(seed)
            with disable_noise_forward(noise_module):
                no_noise = model.decode(mean).squeeze().numpy()
            if not np.isfinite(full).all() or not np.isfinite(no_noise).all():
                raise ValueError("Nonfinite reconstruction")
            save_triplet(out, index, [x, full, no_noise], model.sr)
            results.append({"source": str(path),
                            "full": {"integrity": integrity(full, model.sr),
                                     "quality": spectral_metrics(x, full, model.sr),
                                     "pitch_and_harmonic": compare(x, full, model.sr)},
                            "noise_off": {"integrity": integrity(no_noise, model.sr),
                                          "quality": spectral_metrics(x, no_noise, model.sr),
                                          "pitch_and_harmonic": compare(x, no_noise, model.sr)},
                            "noise_rms_fraction": float(np.sqrt(np.mean((full - no_noise) ** 2)) /
                                                        (np.sqrt(np.mean(full ** 2)) + 1e-12))})
    return {"step": state["global_step"], "sample_rate": model.sr,
            "checkpoint": str(checkpoint), "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--external", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.checkpoint, args.config, args.external, args.out)
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"step": report["step"], "results": [
        {"source": item["source"], "noise_rms_fraction": item["noise_rms_fraction"],
         "full_harmonic_fraction": item["full"]["pitch_and_harmonic"]["output_harmonic_fraction"],
         "noise_off_harmonic_fraction": item["noise_off"]["pitch_and_harmonic"]["output_harmonic_fraction"],
         "full_sc": item["full"]["quality"]["spectral_convergence_mean"],
         "noise_off_sc": item["noise_off"]["quality"]["spectral_convergence_mean"]}
        for item in report["results"]]}, indent=2))
