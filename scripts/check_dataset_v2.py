#!/usr/bin/env python
"""Проверяет разделение песен, жёсткие ссылки и подготовленную LMDB."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml

from prepare_dataset_v2 import sha256_file


def check(manifest_path: Path, processed: Path, verify_hash: bool = True) -> dict:
    from rave.dataset import AudioDataset

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train, holdout = manifest["train"], manifest["holdout"]
    paths = [item["path"] for item in train + holdout]
    hashes = [item["sha256"] for item in train + holdout]
    if not train or not holdout or len(set(paths)) != len(paths) or len(set(hashes)) != len(hashes):
        raise ValueError("Missing songs, duplicate path or duplicate content across splits")
    linked = {p.name for p in (manifest_path.parent / "train").glob("*.mp3")}
    expected = {Path(item["path"]).name for item in train}
    if linked != expected or linked.intersection(Path(item["path"]).name for item in holdout):
        raise ValueError("Train links do not match the frozen train/holdout split")
    for item in train + holdout:
        path = Path(item["path"])
        if verify_hash and sha256_file(path) != item["sha256"]:
            raise ValueError(f"Source changed after manifest creation: {path}")
        if item in train and not os.path.samefile(path, manifest_path.parent / "train" / path.name):
            raise ValueError(f"Training hardlink mismatch: {path}")
    metadata = yaml.safe_load((processed / "metadata.yaml").read_text())
    if metadata["sr"] != 44100 or metadata["channels"] != 1 or metadata["lazy"]:
        raise ValueError(f"Unexpected LMDB metadata: {metadata}")
    audio = AudioDataset(str(processed), n_channels=1)
    samples = [audio[i] for i in (0, len(audio) // 2, len(audio) - 1)]
    if not all(x.shape == (1, 262144) and np.isfinite(x).all() and np.max(np.abs(x)) > 0
               for x in samples):
        raise ValueError("LMDB contains invalid sample crop")
    hours = sum(x["seconds"] for x in train) / 3600
    if abs(metadata["n_seconds"] / 3600 - 2 * hours) > .3:
        raise ValueError("Preprocessed duration inconsistent with source material")
    return {"training_songs": len(train), "holdout_songs": len(holdout),
            "training_hours_unique": round(hours, 3),
            "preprocessed_hours_with_overlap": round(metadata["n_seconds"] / 3600, 3),
            "lmdb_crops": len(audio), "sample_rate": metadata["sr"], "sample_crops_finite": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/v2/manifest.json"))
    parser.add_argument("--processed", type=Path, default=Path("data/v2/processed_2"))
    args = parser.parse_args()
    print(json.dumps(check(args.manifest, args.processed), indent=2))
