#!/usr/bin/env python3
"""Paired video-level bootstrap over seed-averaged identity-test scores."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def auc_rank(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    values, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    positives_by_value = np.bincount(inverse, weights=labels, minlength=len(values))
    n_pos = float(positives_by_value.sum())
    n_neg = float(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    first_ranks = np.cumsum(np.r_[0, counts[:-1]]) + 1
    mean_ranks = first_ranks + (counts - 1) / 2.0
    return float((np.sum(positives_by_value * mean_ranks) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def load_scores(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
    required = {"model", "dataset", "video_id", "y_true", "p_real_mean", "n_seeds"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"Prediction file must contain {sorted(required)}")
    out = defaultdict(dict)
    for r in rows:
        key = (r["model"], r["dataset"])
        vid = r["video_id"]
        if vid in out[key]:
            raise SystemExit(f"Duplicate video row: {key}/{vid}")
        out[key][vid] = (int(r["y_true"]), float(r["p_real_mean"]), int(r["n_seeds"]))
    return out


def run_pair(data, dataset, model_a, model_b, n_boot, rng):
    ka, kb = (model_a, dataset), (model_b, dataset)
    if ka not in data or kb not in data:
        raise SystemExit(f"Missing model predictions for {dataset}: {model_a} or {model_b}")
    if set(data[ka]) != set(data[kb]):
        raise SystemExit(f"The two models do not have identical test video IDs: {model_a} vs {model_b}")
    vids = sorted(data[ka])
    labels_a = np.asarray([data[ka][v][0] for v in vids], dtype=np.int8)
    labels_b = np.asarray([data[kb][v][0] for v in vids], dtype=np.int8)
    if not np.array_equal(labels_a, labels_b):
        raise SystemExit(f"Label mismatch between paired models: {model_a} vs {model_b}")
    seeds_a = {data[ka][v][2] for v in vids}
    seeds_b = {data[kb][v][2] for v in vids}
    if seeds_a != seeds_b or len(seeds_a) != 1:
        raise SystemExit(f"Inconsistent seed counts for {model_a}/{model_b}: {seeds_a} vs {seeds_b}")
    pa = np.asarray([data[ka][v][1] for v in vids], dtype=float)
    pb = np.asarray([data[kb][v][1] for v in vids], dtype=float)
    point_delta = auc_rank(labels_a, pa) - auc_rank(labels_a, pb)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(vids), len(vids))
        if np.unique(labels_a[idx]).size < 2:
            continue
        deltas.append(auc_rank(labels_a[idx], pa[idx]) - auc_rank(labels_a[idx], pb[idx]))
    deltas = np.asarray(deltas, dtype=float)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "dataset": dataset,
        "model_a": model_a,
        "model_b": model_b,
        "n_videos": len(vids),
        "n_seeds": next(iter(seeds_a)),
        "auc_a_seed_mean_scores": auc_rank(labels_a, pa),
        "auc_b_seed_mean_scores": auc_rank(labels_a, pb),
        "point_delta_auc": point_delta,
        "bootstrap_mean_delta_auc": float(np.mean(deltas)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "bootstrap_fraction_delta_positive": float(np.mean(deltas > 0)),
        "bootstrap_replicates_valid": int(len(deltas)),
        "bootstrap_replicates_requested": int(n_boot),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", default="/workspace/predictions/celebdf_identity_seed_mean_video_scores.csv")
    ap.add_argument("--dataset", default="celebdf_identity")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="/workspace/results/celebdf_identity_bootstrap.json")
    ap.add_argument("--comparisons", nargs="+", default=[
        "mfsanet_A4:specxnet", "mfsanet_A4:defax",
        "mfsanet_identity_da:specxnet_identity_da",
        "mfsanet_identity_da:defax_identity_da",
    ], help="Pairs formatted as model_a:model_b")
    args = ap.parse_args()
    data = load_scores(args.predictions)
    rng = np.random.default_rng(args.seed)
    results = []
    for pair in args.comparisons:
        if ":" not in pair:
            raise SystemExit(f"Invalid comparison '{pair}'; expected model_a:model_b")
        a, b = pair.split(":", 1)
        results.append(run_pair(data, args.dataset, a, b, args.n_bootstrap, rng))
    payload = {"protocol": "paired video bootstrap on seed-averaged probabilities",
               "seed": args.seed, "results": results}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
