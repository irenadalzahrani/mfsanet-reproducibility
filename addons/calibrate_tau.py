#!/usr/bin/env python3
"""Validation-based operating threshold per model (video level), with a sensitivity table on the test sets.
For every model/seed: tau* = argmax over tau in [0.05, 0.95] of real-class F1 on the FF++ VALIDATION split (no test data involved);
then accuracy / precision / recall / F1 on the three test sets at tau*, at 0.35 and at 0.5.
    python3 calibrate_tau.py --models specxnet mfsanet_A3 mfsanet_A4 mfsanet_v3 --seeds 42 43 44
Writes results/tau_calibration.csv and results/tau_sensitivity.csv. Read-only on checkpoints. AUC is not affected by tau."""
import argparse, csv, os, sys
sys.path.insert(0, "/workspace")
import numpy as np

GRID = np.round(np.arange(0.05, 0.9501, 0.01), 2)
SENS = (0.30, 0.35, 0.40, 0.45, 0.50)


def metrics_at(scores, labels, thr):
    """Real class (label 1) is the positive class; a video is predicted real if its mean real-class probability >= thr."""
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum()); fp = int(((pred == 1) & (labels == 0)).sum()); fn = int(((pred == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"acc": float((pred == labels).mean()), "precision": prec, "recall": rec, "f1": f1}


def best_tau(scores, labels):
    f1s = np.array([metrics_at(scores, labels, t)["f1"] for t in GRID])
    best = np.flatnonzero(f1s >= f1s.max() - 1e-12)          # ties: take the candidate closest to 0.5
    return float(GRID[best[np.argmin(np.abs(GRID[best] - 0.5))]]), float(f1s.max())


if __name__ == "__main__":
    import torch
    from config import PATHS, TRAIN
    from datasets import DeepfakeFrameDataset
    from eval_predictions import build_registry, predict
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["specxnet", "mfsanet_A3", "mfsanet_A4", "mfsanet_v3"])
    ap.add_argument("--datasets", nargs="+", default=["ffpp", "celebdf", "dfdc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    reg = build_registry(); val_ds = DeepfakeFrameDataset("ffpp", "val", TRAIN.image_size)
    test_ds = {d: DeepfakeFrameDataset(d, "test", TRAIN.image_size) for d in a.datasets}
    cal, sens = [], []
    for tag in a.models:
        for seed in a.seeds:
            ck = os.path.join(PATHS.ckpt_root, tag, f"seed_{seed}", "best.pt")
            if not os.path.isfile(ck): print("[skip] missing", ck); continue
            model = reg[tag]().to(dev); model.load_state_dict(torch.load(ck, map_location=dev)["model_state"]); model.eval()
            _, _, _, vs, vl, _, _ = predict(model, val_ds, dev); vs, vl = np.array(vs), np.array(vl)
            tau, f1v = best_tau(vs, vl)
            print(f"{tag:12s} seed={seed} tau*={tau:.2f} (val F1 {f1v:.4f}; at 0.35: {metrics_at(vs, vl, 0.35)['f1']:.4f}; at 0.50: {metrics_at(vs, vl, 0.5)['f1']:.4f})", flush=True)
            for d, ds in test_ds.items():
                _, _, _, ts, tl, _, _ = predict(model, ds, dev); ts, tl = np.array(ts), np.array(tl)
                for name, thr in (("val_tau", tau), ("fixed_0.35", 0.35), ("fixed_0.50", 0.5)):
                    m = metrics_at(ts, tl, thr); cal.append({"model": tag, "seed": seed, "dataset": d, "rule": name, "tau": thr, "val_tau_star": tau, **m})
                for t in SENS:
                    m = metrics_at(ts, tl, t); sens.append({"model": tag, "seed": seed, "dataset": d, "tau": t, **m})
            del model; torch.cuda.empty_cache()
    os.makedirs(PATHS.results_root, exist_ok=True)
    for fn, rows in (("tau_calibration.csv", cal), ("tau_sensitivity.csv", sens)):
        if rows:
            with open(os.path.join(PATHS.results_root, fn), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("\nSUMMARY: validation-optimal tau per model (mean +- std over seeds) and test F1 (real class) under each rule")
    for tag in a.models:
        ts_ = sorted({r["val_tau_star"] for r in cal if r["model"] == tag and r["dataset"] == a.datasets[0]})
        if not ts_: continue
        per_seed = [r["val_tau_star"] for r in cal if r["model"] == tag and r["dataset"] == a.datasets[0] and r["rule"] == "val_tau"]
        print(f"{tag:12s} tau* = {np.mean(per_seed):.2f} +- {np.std(per_seed):.2f}  (per seed: {per_seed})")
        for d in a.datasets:
            line = []
            for rule in ("val_tau", "fixed_0.35", "fixed_0.50"):
                v = [r["f1"] for r in cal if r["model"] == tag and r["dataset"] == d and r["rule"] == rule]
                line.append(f"{rule}: {100 * np.mean(v):.2f}")
            print(f"     {d:8s} F1  " + " | ".join(line))
    print("saved results/tau_calibration.csv and results/tau_sensitivity.csv")
