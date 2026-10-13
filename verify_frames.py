#!/usr/bin/env python3
"""Count extracted frames and compare with the counts of the original run (from the earlier folder listing).
Usage: python3 verify_frames.py [--dfdc-train-only]"""
import argparse, collections, csv, glob, os, sys
sys.path.insert(0, "/workspace")
from config import PATHS

EXPECTED = {("ffpp", "train", "real"): 14400, ("ffpp", "train", "fake"): 72000, ("ffpp", "val", "real"): 2800, ("ffpp", "val", "fake"): 14000,
            ("ffpp", "test", "real"): 2800, ("ffpp", "test", "fake"): 14000,
            ("celebdf", "train", "real"): 12661, ("celebdf", "train", "fake"): 100880, ("celebdf", "val", "real"): 1560, ("celebdf", "val", "fake"): 5100,
            ("celebdf", "test", "real"): 3560, ("celebdf", "test", "fake"): 6800,
            ("dfdc", "train", "real"): 6760, ("dfdc", "train", "fake"): 62180, ("dfdc", "val", "real"): 4380, ("dfdc", "val", "fake"): 24720,
            ("dfdc", "test", "real"): 3660, ("dfdc", "test", "fake"): 30360}
ap = argparse.ArgumentParser(); ap.add_argument("--dfdc-train-only", action="store_true"); args = ap.parse_args()
problems = 0
for ds in ("ffpp", "celebdf", "dfdc"):
    rows = list(csv.DictReader(open(os.path.join(PATHS.splits_root, f"{ds}_splits.csv"), newline="", encoding="utf-8-sig")))
    for sp in ("train", "val", "test"):
        if ds == "dfdc" and args.dfdc_train_only and sp != "train":
            continue
        for lb in ("real", "fake"):
            d = os.path.join(PATHS.frames_root, ds, sp, lb)
            files = [f for f in os.listdir(d) if f.endswith(".jpg")] if os.path.isdir(d) else []
            per = collections.Counter(f.split("__f")[0] for f in files)
            want_vids = {r["video_id"] for r in rows if r["split"] == sp and r["label"] == lb}
            missing = want_vids - set(per)
            short = sum(1 for v, n in per.items() if n < 20)
            exp = EXPECTED[(ds, sp, lb)]
            dev = abs(len(files) - exp) / exp
            flag = "OK " if (not missing and dev < 0.01) else "CHECK"
            problems += flag != "OK "
            print(f"{flag} {ds:8s} {sp:5s} {lb:4s} frames={len(files):7d} expected~{exp:7d} ({100*dev:.2f}% off) videos={len(per)}/{len(want_vids)} missing={len(missing)} short(<20)={short}")
print("\nRESULT:", "all counts match the original run" if problems == 0 else f"{problems} groups need attention (see CHECK lines)")
