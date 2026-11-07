#!/usr/bin/env python
"""Сравнивает checkpoint на одних и тех же песнях без оценки значимости."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(before: dict, after: dict) -> dict:
    if before["sr"] != after["sr"] or before["evaluation_set"] != after["evaluation_set"]:
        raise ValueError("Different sample rates or evaluation sets")
    a = {str(row["source"]): row for row in before["samples"]}
    b = {str(row["source"]): row for row in after["samples"]}
    if len(a) != len(before["samples"]) or len(b) != len(after["samples"]) or not a or a.keys() != b.keys():
        raise ValueError("Unequal or duplicate source lists")
    result = {"songs": len(a), "before_step": before["step"], "after_step": after["step"]}
    for key in ("spectral_convergence_mean", "log_spectral_distance_mean"):
        diffs = [b[source]["quality_unaligned"][key] - a[source]["quality_unaligned"][key]
                 for source in sorted(a)]
        result[key] = {"mean_after_minus_before": float(np.mean(diffs)),
                       "median_after_minus_before": float(np.median(diffs)),
                       "songs_improved": int(np.sum(np.array(diffs) < 0)),
                       "songs_worsened": int(np.sum(np.array(diffs) > 0))}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(json.loads(args.before.read_text(encoding="utf-8")),
                             json.loads(args.after.read_text(encoding="utf-8"))), indent=2))
