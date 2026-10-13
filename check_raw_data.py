#!/usr/bin/env python3
"""Check that every video listed in the split manifests exists at the path stored in the manifest.
Run on the pod after downloading the datasets:   python3 check_raw_data.py [--dfdc-parts 1 2]
Exit code 0 only if nothing is missing."""
import argparse, collections, csv, os, sys
sys.path.insert(0, "/workspace")
from config import PATHS

ap = argparse.ArgumentParser()
ap.add_argument("--dfdc-parts", nargs="+", type=int, default=[1, 2, 3, 4], help="DFDC archive parts you downloaded (phase A needs only 1 2)")
args = ap.parse_args()
bad_total = 0
for ds in ("ffpp", "celebdf", "dfdc"):
    path = os.path.join(PATHS.splits_root, f"{ds}_splits.csv")
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
    if ds == "dfdc":
        keep = {f"dfdc_train_part_{p}" for p in args.dfdc_parts}
        rows = [r for r in rows if r["source"] in keep]
    missing = [r for r in rows if not os.path.isfile(r["abs_path"])]
    per = collections.Counter((r["source"], r["split"]) for r in rows)
    print(f"\n=== {ds}: {len(rows)} videos expected, {len(missing)} missing")
    for k, n in sorted(per.items()):
        miss = sum(1 for r in missing if (r["source"], r["split"]) == k)
        print(f"   {k[0]:22s} {k[1]:5s} expected={n:5d} missing={miss}")
    for r in missing[:8]:
        print("   MISSING:", r["abs_path"])
    bad_total += len(missing)
print("\nRESULT:", "OK - all videos found" if bad_total == 0 else f"FAILED - {bad_total} videos missing (fix the folder layout before extracting frames)")
sys.exit(0 if bad_total == 0 else 1)
