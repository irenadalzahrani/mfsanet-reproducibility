#!/usr/bin/env python3
"""Inference-time branch-zeroing check for MFSA-Net checkpoints, with per-seed numbers.
For each model/seed/dataset: video-level AUC of the full model, with the spectral branch output set to zero, and with the spatial branch output set to zero.
    python3 zero_branch_check.py --models mfsanet_A3 mfsanet_A4 --datasets ffpp celebdf --seeds 42 43 44
Writes results/branch_zeroing.csv and prints mean +- std over seeds. Read-only on checkpoints."""
import argparse, csv, os, sys
sys.path.insert(0, "/workspace")
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from config import PATHS, TRAIN
from datasets import DeepfakeFrameDataset
from eval_predictions import build_registry, predict

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["mfsanet_A3", "mfsanet_A4"])
ap.add_argument("--datasets", nargs="+", default=["ffpp", "celebdf"])
ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
ap.add_argument("--out", default="branch_zeroing.csv", help="file name under results/")
args = ap.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
reg = build_registry()
rows = []
for ds_name in args.datasets:
    ds = DeepfakeFrameDataset(ds_name, "test", TRAIN.image_size)
    for tag in args.models:
        for seed in args.seeds:
            ck = os.path.join(PATHS.ckpt_root, tag, f"seed_{seed}", "best.pt")
            if not os.path.isfile(ck):
                print("[skip] missing", ck); continue
            model = reg[tag]().to(device); model.load_state_dict(torch.load(ck, map_location=device)["model_state"]); model.eval()
            assert hasattr(model, "spatial") and hasattr(model, "spectral"), "this check needs the full dual-branch model"
            for mode in ("full", "zero_spectral", "zero_spatial"):
                hooks = []
                if mode == "zero_spectral": hooks.append(model.spectral.register_forward_hook(lambda m, i, o: torch.zeros_like(o)))
                if mode == "zero_spatial": hooks.append(model.spatial.register_forward_hook(lambda m, i, o: torch.zeros_like(o)))
                _, _, _, vs, vl, _, _ = predict(model, ds, device)
                for h in hooks: h.remove()
                rows.append({"model": tag, "dataset": ds_name, "seed": seed, "mode": mode, "video_auc": float(roc_auc_score(vl, vs))})
                print(f"{ds_name:8s} {tag:11s} seed={seed} {mode:14s} video AUC={rows[-1]['video_auc']:.4f}", flush=True)
            del model; torch.cuda.empty_cache()
os.makedirs(PATHS.results_root, exist_ok=True)
out = os.path.join(PATHS.results_root, args.out)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "dataset", "seed", "mode", "video_auc"]); w.writeheader(); w.writerows(rows)
print("\nSUMMARY (mean +- std over seeds; drop = full minus zeroed)")
for ds_name in args.datasets:
    for tag in args.models:
        g = {m: [r["video_auc"] for r in rows if r["model"] == tag and r["dataset"] == ds_name and r["mode"] == m] for m in ("full", "zero_spectral", "zero_spatial")}
        if g["full"]:
            print(f"{ds_name:8s} {tag:11s} n={len(g['full'])} | full {np.mean(g['full']):.4f}+-{np.std(g['full']):.4f} | zero-spectral {np.mean(g['zero_spectral']):.4f} (drop {np.mean(g['full'])-np.mean(g['zero_spectral']):+.4f}) | zero-spatial {np.mean(g['zero_spatial']):.4f} (drop {np.mean(g['full'])-np.mean(g['zero_spatial']):+.4f})")
print("saved", out)
