#!/usr/bin/env python
"""Verify that GAN phase actually ran after the shortened warmup.

Only read locally trusted checkpoints (torch.load uses pickle). Usage:
.venv/Scripts/python scripts/check_gan_run.py --run runs/stage1b_early_gan_105k
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def summarize_scalars(events, threshold: int) -> dict:
    summary = {}
    for name in ("adversarial", "loss_dis"):
        values = [(item.step, float(item.value)) for item in events.get(name, [])
                  if item.step >= threshold]
        if not values or not np.isfinite([value for _, value in values]).all():
            raise ValueError(f"No finite {name} scalar measurements after GAN threshold")
        summary[name] = {"count": len(values), "first_step": values[0][0],
                         "last_step": values[-1][0], "first": values[0][1],
                         "last": values[-1][1]}
    return summary


def check(run: Path, threshold: int = 100000) -> dict:
    event_files = list(run.rglob("events.out.tfevents.*"))
    checkpoints = list(run.rglob("*.ckpt"))
    if not event_files or not checkpoints:
        raise FileNotFoundError("No event log or checkpoint in GAN run")
    ckpt = max(checkpoints, key=lambda path: path.stat().st_mtime)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if state["global_step"] <= threshold or state["callbacks"]["WarmupCallback"]["training_steps"] <= threshold:
        raise ValueError("Checkpoint predates the GAN phase")
    nonfinite = sum(int((~torch.isfinite(t)).sum()) for t in state["state_dict"].values()
                    if t.is_floating_point())
    if nonfinite:
        raise ValueError(f"Checkpoint has {nonfinite} nonfinite model parameters")
    events = {}
    for path in event_files:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()
        for name in ("adversarial", "loss_dis"):
            if name in accumulator.Tags()["scalars"]:
                events.setdefault(name, []).extend(accumulator.Scalars(name))
    return {"checkpoint": str(ckpt), "global_step": state["global_step"],
            "nonfinite_parameters": nonfinite,
            "gan_scalars": summarize_scalars(events, threshold)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=100000)
    args = parser.parse_args()
    print(json.dumps(check(args.run, args.threshold), indent=2))
