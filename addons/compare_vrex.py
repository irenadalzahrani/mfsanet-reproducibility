#!/usr/bin/env python3
"""Compare MFSA-Net trained with VREx (mfsanet_vrex) against the plain base model (mfsanet_A3), same seeds. Read-only.  python3 compare_vrex.py"""
import csv, os
import numpy as np
R = "/workspace/results"


def auc_by_seed(path, model):
    out = {}
    if os.path.exists(path):
        for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
            if r["model"] == model and r["level"] == "video" and r["metric"] == "auc" and r["threshold"] == "0.50":
                out[int(r["seed"])] = float(r["value"])
    return out


def zero_drop(path, model, ds):
    rows = [r for r in csv.DictReader(open(path))] if os.path.exists(path) else []
    g = {m: {int(r["seed"]): float(r["video_auc"]) for r in rows if r["model"] == model and r["dataset"] == ds and r["mode"] == m} for m in ("full", "zero_spectral")}
    seeds = sorted(set(g["full"]) & set(g["zero_spectral"]))
    d = np.array([g["full"][s] - g["zero_spectral"][s] for s in seeds])
    return (d.mean(), d.std(), len(seeds)) if len(seeds) else (None, None, 0)


print("1) VIDEO-LEVEL AUC, same seeds (VREx minus base)")
gain = {}
for ds in ("ffpp", "celebdf", "dfdc"):
    b = auc_by_seed(f"{R}/{ds}_metrics_raw_per_seed.csv", "mfsanet_A3"); v = auc_by_seed(f"{R}/{ds}_metrics_raw_per_seed_vrex.csv", "mfsanet_vrex")
    seeds = sorted(set(b) & set(v))
    if not seeds: print(f"   {ds:8s} missing results"); continue
    B = np.array([b[s] for s in seeds]); V = np.array([v[s] for s in seeds]); diff = V.mean() - B.mean(); sd = max(B.std(), V.std())
    gain[ds] = (diff, sd)
    print(f"   {ds:8s} base {B.mean():.4f}+-{B.std():.4f} | VREx {V.mean():.4f}+-{V.std():.4f} | diff {diff:+.4f} (seed std {sd:.4f}) -> {'BEYOND' if abs(diff) > sd else 'within'} seed variability")
print("\n2) SPECTRAL-BRANCH USE: AUC lost when the spectral branch is zeroed at inference (larger = the branch matters more)")
more_use = []
for ds in ("ffpp", "celebdf", "dfdc"):
    a = zero_drop(f"{R}/branch_zeroing.csv", "mfsanet_A3", ds); v = zero_drop(f"{R}/branch_zeroing_vrex.csv", "mfsanet_vrex", ds)
    if a[0] is None or v[0] is None: print(f"   {ds:8s} missing results"); continue
    more_use.append(v[0] - a[0] > max(a[1], v[1]))
    print(f"   {ds:8s} base drop {a[0]:+.4f}+-{a[1]:.4f} | VREx drop {v[0]:+.4f}+-{v[1]:.4f}")
print("\n3) VERDICT (criteria fixed before the experiment)")
helps = any(gain.get(d, (0, 1))[0] > gain.get(d, (0, 1))[1] for d in ("celebdf", "dfdc")) and gain.get("ffpp", (0, 1))[0] > -gain.get("ffpp", (0, 1))[1]
print("   (a) AUC improves on CelebDF or DFDC beyond seed variability, without a loss on FF++:", "YES" if helps else "NO")
print("   (b) the spectral branch matters clearly more than in the base model on at least one set:", "YES" if any(more_use) else "NO")
print("   => " + ("VREx is a candidate improvement: report with the per-seed numbers." if helps and any(more_use) else "report as a negative or inconclusive result (the five earlier attempts are the same kind of evidence)."))
