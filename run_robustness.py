#!/usr/bin/env python3
"""JPEG / Gaussian-noise robustness sweep (9 conditions) for the four models of the paper on the three test sets.
    python3 run_robustness.py --tags specxnet defax mfsanet_A3 mfsanet_A4 --datasets ffpp celebdf dfdc --seeds 42 43 44
Writes results/robustness_all_combined.csv (model,dataset,seed,condition,auc,acc) and results/robustness_averaged.csv (adds n_seeds).
Gaussian noise uses a fixed NumPy seed (1234) before every model/dataset sweep so the sweep is reproducible. Restartable: finished sweeps are skipped."""
import argparse, csv, os, shutil, sys, collections
sys.path.insert(0, "/workspace")
import numpy as np, torch
from config import PATHS
from robustness_common import run_robustness_sweep
from eval_predictions import build_registry

ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", default=["specxnet", "defax", "mfsanet_A3", "mfsanet_A4"])
ap.add_argument("--datasets", nargs="+", default=["ffpp", "celebdf", "dfdc"])
ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
args = ap.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
registry = build_registry()
os.makedirs(PATHS.results_root, exist_ok=True)
for tag in args.tags:
    for ds in args.datasets:
        out = os.path.join(PATHS.results_root, f"{tag}_{ds}_robustness_sweep.csv")
        if os.path.exists(out):
            print(f"[skip] {out}", flush=True); continue
        np.random.seed(1234); torch.manual_seed(1234)
        run_robustness_sweep(tag, registry[tag], args.seeds, ds, device)
        shutil.move(os.path.join(PATHS.results_root, f"{tag}_robustness_sweep.csv"), out)
rows = []
for tag in args.tags:
    for ds in args.datasets:
        p = os.path.join(PATHS.results_root, f"{tag}_{ds}_robustness_sweep.csv")
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                rows.append({"model": tag, "dataset": ds, "seed": r["seed"], "condition": r["condition"], "auc": r["auc"], "acc": r["acc"]})
with open(os.path.join(PATHS.results_root, "robustness_all_combined.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "dataset", "seed", "condition", "auc", "acc"]); w.writeheader(); w.writerows(rows)
g = collections.defaultdict(list)
for r in rows:
    g[(r["model"], r["dataset"], r["condition"])].append(float(r["auc"]))
with open(os.path.join(PATHS.results_root, "robustness_averaged.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "dataset", "condition", "mean_auc", "n_seeds"])
    for (m, d, c), v in sorted(g.items()):
        w.writerow([m, d, c, f"{np.mean(v):.6f}", len(v)])
print("wrote robustness_all_combined.csv and robustness_averaged.csv; cells with fewer than 3 seeds:", sum(1 for v in g.values() if len(v) < 3))
