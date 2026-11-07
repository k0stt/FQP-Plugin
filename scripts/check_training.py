#!/usr/bin/env python
"""Check a preprocessed RAVE dataset and optionally a saved training checkpoint.

Usage: .venv/Scripts/python scripts/check_training.py --checkpoint runs/.../best.ckpt
Only load checkpoints from trusted local runs (torch serialization uses pickle).
"""
import argparse
import json
from pathlib import Path

import gin
import numpy as np
import soundfile as sf
import torch
import yaml


def audit_data(root: Path) -> dict:
    metadata = yaml.safe_load((root / "processed/metadata.yaml").read_text())
    manifest = (root / "manifest.txt").read_text(encoding="utf-8")
    names = [line.strip() for line in manifest.splitlines()
             if line.strip().endswith(".mp3") and not line.startswith("**")]
    raw = {p.name: p for p in (root / "raw/openlofi").glob("*.mp3")}
    if len(names) != len(set(names)) or not names or set(names) != set(raw):
        raise ValueError(f"Manifest/raw mismatch: missing={set(names)-set(raw)}, extra={set(raw)-set(names)}")
    duration = 0.0
    for name in names:
        info = sf.info(str(raw[name]))
        if info.duration <= 0 or info.samplerate <= 0 or info.channels <= 0:
            raise ValueError(f"Invalid audio: {name}")
        duration += info.duration
    if metadata["sr"] != 44100 or metadata["channels"] != 1 or metadata["n_seconds"] <= 0:
        raise ValueError(f"Unexpected preprocessed metadata: {metadata}")
    if not (root / "processed/data.mdb").stat().st_size:
        raise ValueError("Empty preprocessed LMDB")
    return {"files": len(names), "raw_hours": round(duration / 3600, 3),
            "processed_hours": round(metadata["n_seconds"] / 3600, 3),
            "sample_rate": metadata["sr"], "channels": metadata["channels"]}


def audit_checkpoint(path: Path, config: Path) -> dict:
    import rave

    gin.parse_config_file(str(config))
    model = rave.RAVE(n_channels=1)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    sr = model.sr
    # Short raw passage for an inference smoke test; NOT a held-out quality set.
    raw_file = sorted(Path("data/raw/openlofi").glob("*.mp3"))[0]
    import soxr
    samples, source_sr = sf.read(str(raw_file), frames=int(sf.info(str(raw_file)).samplerate * 3),
                                 dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if source_sr != sr:
        samples = soxr.resample(samples, source_sr, sr)
    x = torch.from_numpy(samples[:sr * 2].copy()).reshape(1, 1, -1)
    with torch.no_grad():
        z = model.encode(x)
        mean = z[:, :z.shape[1] // 2]
        y = model.decode(mean)
    output = y.squeeze().numpy()
    if not np.isfinite(output).all() or len(output) < sr or np.max(np.abs(output)) == 0:
        raise ValueError("Checkpoint reconstruction is silent, too short or nonfinite")
    return {"epoch": ckpt["epoch"], "global_step": ckpt["global_step"],
            "latent_shape": list(mean.shape), "output_samples": len(output),
            "peak": float(np.max(np.abs(output))),
            "rms": float(np.sqrt(np.mean(output ** 2))), "sample_rate": sr}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/lofi_v2_nwn.gin"))
    args = parser.parse_args()
    report = {"data": audit_data(Path("data"))}
    if args.checkpoint:
        report["checkpoint"] = audit_checkpoint(args.checkpoint, args.config)
    print(json.dumps(report, indent=2))
