#!/usr/bin/env python
"""Deterministic checkpoint reconstruction on the fixed RAVE validation split.

The split is by preprocessed crop (seed 42), NOT by song: these scores do not
measure generalization to unseen songs. Evaluates the un-exported 128-D model.
Only load trusted local checkpoints (torch.load uses pickle).
"""
import argparse
import json
from pathlib import Path

import gin
import numpy as np
import torch
from scipy.signal import correlate, correlation_lags

from rave_reconstruct import integrity, spectral_metrics


def align_signals(x: np.ndarray, y: np.ndarray, sr: int):
    """Positive lag means the output y is delayed relative to input x."""
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    lag_limit = min(n - 1, sr // 2)
    c = correlate(y, x, mode="full", method="fft")
    lags = correlation_lags(n, n, mode="full")
    region = np.abs(lags) <= lag_limit
    lag = int(lags[region][np.argmax(np.abs(c[region]))])
    if lag >= 0:
        x, y = x[:n - lag], y[lag:]
    else:
        x, y = x[-lag:], y[:n + lag]
    return x, y, lag


def evaluate(checkpoint: Path, config: Path, count: int) -> dict:
    import rave
    from rave import dataset

    torch.manual_seed(111)
    gin.parse_config_file(str(config))
    model = rave.RAVE(n_channels=1)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    # Training split is seeded inside RAVE; use unaugmented raw crops for evaluation.
    raw = dataset.AudioDataset("data/processed", n_channels=1)
    _, val = dataset.split_dataset(raw, 98)
    if count <= 0 or count > len(val):
        raise ValueError(f"count must be in [1, {len(val)}]")
    rows = []
    with torch.inference_mode():
        for i in range(count):
            x = val[i]
            # Center crop of identical length on every rerun; model context > 200ms.
            n = 131072
            if x.shape[-1] < n:
                raise ValueError(f"Validation crop {i} shorter than {n}")
            start = (x.shape[-1] - n) // 2
            reference = x[0, start:start + n]
            z = model.encode(torch.from_numpy(reference.copy()).reshape(1, 1, -1))
            mean = z[:, :z.shape[1] // 2]
            y = model.decode(mean).squeeze().numpy()
            check = integrity(y, model.sr)
            if check["n_nonfinite"] or check["rms"] <= 0:
                raise ValueError(f"Invalid reconstruction {i}: {check}")
            metrics = spectral_metrics(reference, y, model.sr)
            aligned_x, aligned_y, lag = align_signals(reference, y, model.sr)
            aligned = spectral_metrics(aligned_x, aligned_y, model.sr)
            correlation = float(np.dot(aligned_x.astype(np.float64), aligned_y.astype(np.float64)) /
                                (np.linalg.norm(aligned_x) * np.linalg.norm(aligned_y) + 1e-12))
            rows.append({"validation_index": int(val.indices[i]), "latent_frames": mean.shape[-1],
                         "output_samples": len(y), "integrity": check, "quality_unaligned": metrics,
                         "lag_samples_cross_correlation": lag, "aligned_correlation": correlation,
                         "quality_aligned": aligned})
    return {"epoch": state["epoch"], "step": state["global_step"], "sr": model.sr,
            "latent_channels": model.latent_size, "samples": rows,
            "mean_spectral_convergence_unaligned": float(np.mean([
                r["quality_unaligned"]["spectral_convergence_mean"] for r in rows])),
            "mean_lsd_unaligned": float(np.mean([
                r["quality_unaligned"]["log_spectral_distance_mean"] for r in rows])),
            "mean_spectral_convergence_aligned": float(np.mean([
                r["quality_aligned"]["spectral_convergence_mean"] for r in rows])),
            "mean_lsd_aligned": float(np.mean([
                r["quality_aligned"]["log_spectral_distance_mean"] for r in rows]))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/lofi_v2_nwn.gin"))
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("outputs/stage1b/evaluation.json"))
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.config, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))
