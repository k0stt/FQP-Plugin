#!/usr/bin/env python
"""Готовит отдельный опыт с более ранним запуском GAN.

Копирует checkpoint и меняет только длительность первого этапа в Gin;
остальное состояние обучения сохраняется. Загружать только доверенные веса.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch


DEFAULT_SOURCE = Path("runs/stage1b_100000_from50k/lofi_v2_nwn_resume_17211b00e7"
                      "/version_0/checkpoints/best.ckpt")
OLD_BINDING = "model.RAVE.phase_1_duration = 1000000"
NEW_BINDING = "model.RAVE.phase_1_duration = 100000"


def digest(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hash_.update(chunk)
    return hash_.hexdigest()


def fork_config(original: str) -> str:
    lines = original.splitlines(keepends=True)
    if sum(line.strip() == OLD_BINDING for line in lines) != 1 or any(
            line.strip() == NEW_BINDING for line in lines):
        raise ValueError("Expected exactly one original 1,000,000-step phase-1 binding")
    return "".join(line.replace(OLD_BINDING, NEW_BINDING) if line.strip() == OLD_BINDING
                   else line for line in lines)


def prepare(source: Path, target: Path) -> dict:
    source_config = source.parent.parent.parent / "config.gin"
    text = fork_config(source_config.read_text(encoding="utf-8"))
    state = torch.load(source, map_location="cpu", weights_only=False)
    step = state["global_step"]
    warmup = state["callbacks"]["WarmupCallback"]["training_steps"]
    scheduler_iters = state["lr_schedulers"][0]["total_iters"]
    if step != warmup or step >= 100000 or scheduler_iters != 1000000:
        raise ValueError("Unexpected source step/warmup/LR schedule; do not fork silently")
    ckpt_target = target / "checkpoints" / "seed.ckpt"
    config_target = target / "config.gin"
    if ckpt_target.exists() or config_target.exists():
        raise FileExistsError(f"Experimental seed already exists: {target}")
    ckpt_target.parent.mkdir(parents=True, exist_ok=True)
    config_target.write_bytes(text.encode("utf-8"))
    shutil.copyfile(source, ckpt_target)
    report = {"source_step": step, "gan_start_step": 100000,
              "lr_schedule_total_iters_unchanged": scheduler_iters,
              "source_checkpoint_sha256": digest(source),
              "seed_checkpoint_sha256": digest(ckpt_target),
              "config_sha256": digest(config_target),
              "seed_checkpoint": str(ckpt_target)}
    (target / "provenance.json").write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=Path("runs/stage1b_early_gan_seed"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.target), indent=2))
