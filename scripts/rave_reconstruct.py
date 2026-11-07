#!/usr/bin/env python
"""Реконструирует аудио предобученной RAVE и сохраняет WAV с отчётом.

Для сравнения используется среднее постериора; задержка на импульсе
измеряется отдельно, в новом процессе.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SEED = 111
MODEL_PATH = Path("models/musicnet.ts")


def load_audio(path: Path, sr_target: int) -> tuple[np.ndarray, int]:
    """Переводит аудио в моно и частоту модели."""
    x, sr = sf.read(str(path), always_2d=True, dtype="float32")
    x = x.mean(axis=1)
    if sr != sr_target:
        import soxr
        x = soxr.resample(x, sr, sr_target, quality="HQ")
        sr = sr_target
    return x, sr


def save_wav(path: Path, y: np.ndarray, sr: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(y, -1.0, 1.0), sr, subtype="PCM_24")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spectral_metrics(x: np.ndarray, y: np.ndarray, sr: int) -> dict:
    """Считает спектральные расстояния на трёх размерах окна."""
    n = min(len(x), len(y))
    x, y = x[:n].astype(np.float64), y[:n].astype(np.float64)
    out = {}
    sc_all, lsd_all = [], []
    for fft in (256, 1024, 4096):
        hop = fft // 2
        win = np.hanning(fft)
        frames = 1 + (n - fft) // hop
        if frames < 2:
            continue
        idx = np.arange(fft)[None, :] + hop * np.arange(frames)[:, None]
        X = np.abs(np.fft.rfft(x[idx] * win, axis=1))
        Y = np.abs(np.fft.rfft(y[idx] * win, axis=1))
        sc = np.linalg.norm(X - Y) / (np.linalg.norm(X) + 1e-12)
        lsd = np.sqrt(np.mean((np.log1p(X) - np.log1p(Y)) ** 2))
        sc_all.append(sc)
        lsd_all.append(lsd)
        out[f"spectral_convergence_{fft}"] = round(float(sc), 6)
        out[f"log_spectral_distance_{fft}"] = round(float(lsd), 6)
    out["spectral_convergence_mean"] = round(float(np.mean(sc_all)), 6)
    out["log_spectral_distance_mean"] = round(float(np.mean(lsd_all)), 6)
    return out


def integrity(y: np.ndarray, sr: int) -> dict:
    """Проверяет пики, неконечные значения и длинные участки тишины."""
    finite = np.isfinite(y)
    absy = np.abs(y)
    thr = 1e-4
    quiet = absy < thr
    min_len = int(0.020 * sr)
    runs, cur = [], 0
    for q in quiet:
        cur = cur + 1 if q else 0
        if cur >= min_len:
            runs.append(cur)
    edge = quiet[:min_len].all(), quiet[-min_len:].all()
    body_breaks = 0
    in_silence = False
    start = 0
    for i, q in enumerate(quiet):
        if q and not in_silence:
            start, in_silence = i, True
        elif not q and in_silence:
            in_silence = False
            if start > 0 and i - start >= min_len:
                body_breaks += 1
    return {
        "n_nonfinite": int((~finite).sum()),
        "n_clipped": int((absy > 0.999).sum()),
        "peak": round(float(absy[finite].max()) if finite.any() else 0.0, 6),
        "rms": round(float(np.sqrt(np.mean(y[finite] ** 2))) if finite.any() else 0.0, 6),
        "internal_silence_breaks": body_breaks,
        "starts_with_silence": bool(edge[0]),
        "ends_with_silence": bool(edge[1]),
    }


def impulse_latency(model, device, sr: int) -> dict:
    """Измеряет задержку по положению пика после подачи импульса."""
    n = 2 * sr
    imp = torch.zeros(1, 1, n)
    pos = n // 4
    imp[0, 0, pos] = 1.0
    xt = imp.to(device)
    with torch.no_grad():
        z = encode_deterministic(model, xt)
        yi = model.decode(z, True).detach().cpu().squeeze().numpy()
    peak = int(np.argmax(np.abs(yi)))
    return {
        "impulse_position": pos,
        "output_peak_position": peak,
        "delay_samples": peak - pos,
        "delay_ms": round((peak - pos) / sr * 1000, 2),
        "peak_amplitude": round(float(np.abs(yi).max()), 4),
    }


def encode_deterministic(model, xt: torch.Tensor) -> torch.Tensor:
    """Берёт среднее постериора вместо случайного сэмпла энкодера."""
    bs = xt.shape[:-2]
    xp = model.pqmf(xt.reshape(-1, 1, xt.shape[-1]))
    xp = xp.reshape(bs + (-1, xp.shape[-1]))
    raw = model.encoder(xp)
    mean, _scale = raw.chunk(2, 1)
    z = mean - model.latent_mean.reshape(1, -1, 1)
    z = torch.nn.functional.conv1d(z, model.latent_pca.reshape(128, 128, 1))
    return z[:, :model.latent_size]


def reconstruct(model, x: np.ndarray, device: str):
    """Измеряет время кодирования и декодирования отдельно."""
    xt = torch.from_numpy(x).reshape(1, 1, -1).to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        z = encode_deterministic(model, xt)
    if device == "cuda":
        torch.cuda.synchronize()
    t_enc = time.perf_counter() - t0
    t0 = time.perf_counter()
    with torch.no_grad():
        y = model.decode(z, True)
    if device == "cuda":
        torch.cuda.synchronize()
    t_dec = time.perf_counter() - t0
    return y.cpu().squeeze().numpy(), t_enc, t_dec, tuple(z.shape)


def main():
    ap = argparse.ArgumentParser(description="E0: RAVE reconstruction baseline")
    ap.add_argument("--input", default=None, help="аудиофайл (wav/flac/ogg)")
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    choices=["cpu", "cuda"])
    ap.add_argument("--out-dir", default="outputs/e0")
    ap.add_argument("--latency-only", action="store_true",
                    help="только замер задержки на импульсе (свежий процесс)")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    t0 = time.perf_counter()
    model = torch.jit.load(args.model).eval().to(args.device)
    load_s = time.perf_counter() - t0
    sr = int(model.sr)

    if args.latency_only:
        print(json.dumps({"latency": impulse_latency(model, args.device, sr)},
                         ensure_ascii=False, indent=2))
        return

    if args.input is None:
        ap.error("--input обязателен без флага --latency-only")
    inp = Path(args.input)
    x, sr = load_audio(inp, sr)
    dur = len(x) / sr

    y, t_enc, t_dec, z_shape = reconstruct(model, x, args.device)
    total_s = t_enc + t_dec

    # Для аудиофайла ищем сдвиг по кросс-корреляции, для импульса — отдельно.
    n = min(len(x), len(y))
    L = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.fft.rfft(x[:n]) * np.conj(np.fft.rfft(y[:n])), L)
    lag = int(np.argmax(np.abs(c[: int(0.5 * sr)])))
    corr = float(c[lag] / (np.linalg.norm(x[:n]) * np.linalg.norm(y[:n]) + 1e-12))

    import subprocess
    lat = subprocess.run(
        [sys.executable, __file__, "--latency-only",
         "--model", args.model, "--device", args.device],
        capture_output=True, text=True, check=True,
    )
    latency = json.loads(lat.stdout)["latency"]
    latency["group_delay_samples"] = lag
    latency["group_delay_ms"] = round(lag / sr * 1000, 2)
    latency["group_delay_corr"] = round(corr, 4)

    out_dir = Path(args.out_dir) / inp.stem
    wav_sha = save_wav(out_dir / "reconstruction.wav", y, sr)

    report = {
        "stage": "E0 (Этап 1: baseline reconstruction)",
        "input": str(inp),
        "input_duration_s": round(dur, 3),
        "model": {
            "path": args.model,
            "sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
            "sample_rate": sr,
            "latent_shape": list(z_shape),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "cpu_threads": torch.get_num_threads(),
            "platform": platform.platform(),
        },
        "seed": SEED,
        "determinism_protocol": "encode uses posterior MEAN (no randn sampling) -> fully deterministic; stock model.encode() samples VAE posterior (aten::randn) and needs fresh process + seed",
        "timing": {
            "model_load_s": round(load_s, 3),
            "encode_s": round(t_enc, 4),
            "decode_s": round(t_dec, 4),
            "total_s": round(total_s, 4),
            "rtf": round(total_s / dur, 5),
        },
        "latency": latency,
        "integrity": integrity(y, sr),
        "quality_vs_input": spectral_metrics(x, y, sr),
        "output_wav": str(out_dir / "reconstruction.wav"),
        "output_wav_sha256": wav_sha,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
