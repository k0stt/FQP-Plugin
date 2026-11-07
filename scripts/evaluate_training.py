#!/usr/bin/env python
"""Оценивает checkpoint на фиксированных фрагментах или внешних файлах.

Внутренняя выборка разделена по фрагментам, а не по песням. Checkpoint
загружается через pickle; использовать только доверенные файлы.
"""
import argparse
import hashlib
import json
from pathlib import Path

import gin
import numpy as np
import soundfile as sf
import torch
from scipy.signal import correlate, correlation_lags

from rave_reconstruct import encode_deterministic, integrity, spectral_metrics


def align_signals(x: np.ndarray, y: np.ndarray, sr: int):
    """Положительный сдвиг означает, что выход запаздывает относительно входа."""
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


def listening_pair(x: np.ndarray, y: np.ndarray, sr: int, fade_ms: float):
    """Применяет общий gain и одинаковый fade; метрики считаются до обработки."""
    if fade_ms < 0 or fade_ms * sr / 1000 >= min(len(x), len(y)) / 2:
        raise ValueError("Fade must be nonnegative and shorter than half a crop")
    gain = min(1.0, 0.9 / max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), 1e-12))
    n = int(fade_ms * sr / 1000)
    def apply(a):
        result = a.copy() * gain
        if n:
            envelope = np.sin(np.linspace(0, np.pi / 2, n)) ** 2
            result[:n] *= envelope
            result[-n:] *= envelope[::-1]
        return result
    return apply(x), apply(y)


def holdout_files(manifest: dict) -> list[Path]:
    train = manifest["train"]
    holdout = manifest["holdout"]
    if not holdout or set(item["path"] for item in train) & set(
            item["path"] for item in holdout) or set(item["sha256"] for item in train) & set(
            item["sha256"] for item in holdout):
        raise ValueError("Missing holdout or train/holdout song overlap")
    return [Path(item["path"]) for item in holdout]


def evaluate(checkpoint: Path, config: Path, count: int, reference_model: Path = None,
             audio_dir: Path = None, external_files=None, fade_ms: float = 0) -> dict:
    import rave
    from rave import dataset

    torch.manual_seed(111)
    gin.parse_config_file(str(config))
    model = rave.RAVE(n_channels=1)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    baseline = torch.jit.load(str(reference_model), map_location="cpu").eval() if reference_model else None
    if baseline is not None and int(baseline.sr) != model.sr:
        raise ValueError("Reference model sample rate differs from training model")
    if external_files:
        import soxr
        examples = []
        for path in external_files:
            info = sf.info(str(path))
            start = max(0, (info.frames - int(3 * info.samplerate)) // 2)
            x, source_sr = sf.read(str(path), start=start, frames=int(3 * info.samplerate),
                                   always_2d=True, dtype="float32")
            x = x.mean(axis=1)
            if source_sr != model.sr:
                x = soxr.resample(x, source_sr, model.sr)
            if len(x) < 131072:
                raise ValueError(f"External file too short: {path}")
            examples.append((str(path), x))
    else:
        # Внутреннее разделение RAVE не проверяет новые песни.
        raw = dataset.AudioDataset("data/processed", n_channels=1)
        _, val = dataset.split_dataset(raw, 98)
        if count <= 0 or count > len(val):
            raise ValueError(f"count must be in [1, {len(val)}]")
        examples = [(int(val.indices[i]), val[i][0]) for i in range(count)]
    rows = []
    with torch.inference_mode():
        for i, (source, x) in enumerate(examples):
            seed = 111 + (source if isinstance(source, int) else i)
            # Берём центральный фрагмент одинаковой длины.
            n = 131072
            if x.shape[-1] < n:
                raise ValueError(f"Validation crop {i} shorter than {n}")
            start = (x.shape[-1] - n) // 2
            reference = x[start:start + n]
            xt = torch.from_numpy(reference.copy()).reshape(1, 1, -1)
            z = model.encode(xt)
            mean = z[:, :z.shape[1] // 2]
            # Шумовая ветвь использует случайные значения даже в eval().
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                y = model.decode(mean).squeeze().numpy()
            check = integrity(y, model.sr)
            if check["n_nonfinite"] or check["rms"] <= 0:
                raise ValueError(f"Invalid reconstruction {i}: {check}")
            if audio_dir is not None:
                audio_dir.mkdir(parents=True, exist_ok=True)
                listen_x, listen_y = listening_pair(reference, y, model.sr, fade_ms)
                sf.write(str(audio_dir / f"crop_{i:02d}_input.wav"), listen_x,
                         model.sr, subtype="FLOAT")
                sf.write(str(audio_dir / f"crop_{i:02d}_reconstruction.wav"), listen_y,
                         model.sr, subtype="FLOAT")
            metrics = spectral_metrics(reference, y, model.sr)
            aligned_x, aligned_y, lag = align_signals(reference, y, model.sr)
            aligned = spectral_metrics(aligned_x, aligned_y, model.sr)
            correlation = float(np.dot(aligned_x.astype(np.float64), aligned_y.astype(np.float64)) /
                                (np.linalg.norm(aligned_x) * np.linalg.norm(aligned_y) + 1e-12))
            row = {"source": source, "latent_frames": mean.shape[-1],
                         "output_samples": len(y), "integrity": check, "quality_unaligned": metrics,
                         "lag_samples_cross_correlation": lag, "aligned_correlation": correlation,
                         "quality_aligned": aligned}
            if baseline is not None:
                baseline_z = encode_deterministic(baseline, xt)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    baseline_y = baseline.decode(baseline_z, True).squeeze().numpy()
                row["pretrained"] = {
                    "latent_shape": list(baseline_z.shape), "integrity": integrity(baseline_y, model.sr),
                    "quality_unaligned": spectral_metrics(reference, baseline_y, model.sr)}
            rows.append(row)
    report = {"epoch": state["epoch"], "step": state["global_step"], "sr": model.sr,
            "evaluation_set": "external_files" if external_files else "rave_crop_validation",
            "latent_channels": model.latent_size, "samples": rows,
            "nonfinite_samples": sum(r["integrity"]["n_nonfinite"] for r in rows),
            "overrange_crops": sum(r["integrity"]["peak"] > 1 for r in rows),
            "mean_spectral_convergence_unaligned": float(np.mean([
                r["quality_unaligned"]["spectral_convergence_mean"] for r in rows])),
            "mean_lsd_unaligned": float(np.mean([
                r["quality_unaligned"]["log_spectral_distance_mean"] for r in rows])),
            "mean_spectral_convergence_aligned": float(np.mean([
                r["quality_aligned"]["spectral_convergence_mean"] for r in rows])),
            "mean_lsd_aligned": float(np.mean([
                r["quality_aligned"]["log_spectral_distance_mean"] for r in rows]))}
    if baseline is not None:
        report["pretrained_model"] = str(reference_model)
        report["pretrained_mean_sc_unaligned"] = float(np.mean([
            r["pretrained"]["quality_unaligned"]["spectral_convergence_mean"] for r in rows]))
        report["pretrained_mean_lsd_unaligned"] = float(np.mean([
            r["pretrained"]["quality_unaligned"]["log_spectral_distance_mean"] for r in rows]))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/lofi_v2_nwn.gin"))
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--reference-model", type=Path, help="optional pretrained TorchScript baseline")
    parser.add_argument("--audio-dir", type=Path, help="write matching gain input/output WAV pairs")
    parser.add_argument("--fade-ms", type=float, default=0,
                        help="optional identical edge fade for listening WAVs only")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--external", type=Path, nargs="+", help="independent audio files (center 3 s)")
    source.add_argument("--holdout-manifest", type=Path,
                        help="evaluate every song in the v2 holdout manifest (center 3 s)")
    parser.add_argument("--output", type=Path, default=Path("outputs/stage1b/evaluation.json"))
    args = parser.parse_args()
    external = args.external
    if args.holdout_manifest:
        manifest_bytes = args.holdout_manifest.read_bytes()
        manifest = json.loads(manifest_bytes)
        external = holdout_files(manifest)
    result = evaluate(args.checkpoint, args.config, args.count, args.reference_model,
                      args.audio_dir, external, args.fade_ms)
    if args.holdout_manifest:
        result["evaluation_set"] = "song_disjoint_holdout_v2"
        result["holdout_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))
