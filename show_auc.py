#!/usr/bin/env python3
"""Print video-level AUC per model and seed from results/<dataset>_metrics_raw_per_seed.csv, next to the old reference run.
Usage: python3 show_auc.py ffpp celebdf"""
import csv, collections, os, sys
import numpy as np
REF = "/workspace/reference_results"
for ds in sys.argv[1:]:
    new = collections.defaultdict(dict)
    p = f"/workspace/results/{ds}_metrics_raw_per_seed.csv"
    if not os.path.exists(p):
        print(f"{ds}: {p} not found (run eval_predictions.py --dataset {ds} first)"); continue
    for r in csv.DictReader(open(p)):
        if r["level"] == "video" and r["metric"] == "auc" and r["threshold"] == "0.35":
            new[r["model"]][int(r["seed"])] = float(r["value"])
    for m, d in sorted(new.items()):
        old = {}
        rp = f"{REF}/{m}_raw_per_seed.csv"
        if os.path.exists(rp):
            for r in csv.DictReader(open(rp)):
                if r["dataset"] == ds and r["level"] == "video" and r["metric"] == "auc" and r["threshold"] == "calibrated_0.35":
                    old[int(r["seed"])] = float(r["value"])
        o = f"old run: mean {np.mean(list(old.values())):.4f} (seeds {', '.join(f'{k}:{v:.4f}' for k, v in sorted(old.items()))})" if old else "no old reference (new experiment)"
        print(f"{ds:16s} {m:22s} NEW: " + ", ".join(f"{k}:{v:.4f}" for k, v in sorted(d.items())) + f" | {o}")
