#!/usr/bin/env python
"""Freeze a song-disjoint, content-fingerprinted RAVE v2 corpus.

Only metadata is written to data/v2/manifest.json. With --link-training,
hardlink training MP3s into data/v2/train for rave preprocess; original files
are never moved or overwritten. Holdout songs are NOT linked into that folder.
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import soundfile as sf


SOURCES = (Path("data/raw/openlofi"), Path("data/raw/openlofi_v2"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_tracks(original: list, added: list, seed: int = 42, holdout_count: int = 12):
    if not original or len(added) <= holdout_count or holdout_count < 1:
        raise ValueError("Insufficient songs for a separate validation holdout")
    indices = list(range(len(added)))
    random.Random(seed).shuffle(indices)
    held = set(indices[:holdout_count])
    train = sorted(original + [item for i, item in enumerate(added) if i not in held],
                   key=lambda item: item["path"])
    holdout = sorted([item for i, item in enumerate(added) if i in held],
                     key=lambda item: item["path"])
    assert set(item["path"] for item in train).isdisjoint(item["path"] for item in holdout)
    return train, holdout


def inventory(source: Path) -> list:
    if not source.is_dir():
        raise FileNotFoundError(source)
    result = []
    for path in sorted(source.glob("*.mp3")):
        info = sf.info(str(path))
        if info.duration < 4 or info.channels < 1 or info.samplerate < 22050:
            raise ValueError(f"Invalid or short source: {path}")
        result.append({"path": path.as_posix(), "sha256": sha256_file(path),
                       "seconds": round(info.duration, 3), "sample_rate": info.samplerate,
                       "channels": info.channels, "bytes": path.stat().st_size})
    return result


def prepare(output: Path, link_training: bool = False) -> dict:
    original, added = [inventory(folder) for folder in SOURCES]
    all_files = original + added
    if len({item["sha256"] for item in all_files}) != len(all_files):
        raise ValueError("Duplicate content in original/new corpus; inspect before splitting")
    if len({Path(item["path"]).name for item in all_files}) != len(all_files):
        raise ValueError("Duplicate basenames prevent unambiguous hardlinks")
    train, holdout = split_tracks(original, added)
    document = {"version": 2, "seed": 42, "split_unit": "song/file (not LMDB crop)",
                "license_claim": "CC0 1.0 Universal; user-supplied repository declaration, source URL not verified",
                "train": train, "holdout": holdout}
    encoded = json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"Frozen manifest differs: {output}. Use a NEW version path.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        output.write_bytes(encoded.encode("utf-8"))
    if link_training:
        target = output.parent / "train"
        target.mkdir(exist_ok=True)
        for item in train:
            source = Path(item["path"])
            link = target / source.name
            if link.exists() and not os.path.samefile(source, link):
                raise FileExistsError(f"Existing nonmatching train file: {link}")
        for item in train:
            source = Path(item["path"])
            link = target / source.name
            if not link.exists():
                os.link(source, link)
    return {"train_files": len(train), "holdout_files": len(holdout),
            "train_hours": round(sum(x["seconds"] for x in train) / 3600, 3),
            "holdout_hours": round(sum(x["seconds"] for x in holdout) / 3600, 3),
            "manifest_sha256": sha256_file(output),
            "manifest_path": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/v2/manifest.json"))
    parser.add_argument("--link-training", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.link_training), indent=2))
