#!/usr/bin/env python3
"""Build a separate CelebDF identity-disjoint train/test manifest.

The existing official video-level split is never modified. This supplementary
identity protocol uses only rows currently in official train/val; official-test
rows are omitted from this alternate manifest. Celeb-real/fake names must encode
one/two subject IDs as idN or idN_idM. Unidentified YouTube-real rows and fake
pairs crossing the train/test identity boundary are excluded and logged.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ID_RE = re.compile(r"id\d+")
REQUIRED = {"video_id", "split", "label", "abs_path", "source"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="splits/celebdf_splits.csv")
    ap.add_argument("--output", default="splits/celebdf_identity_splits.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-fraction", type=float, default=0.20)
    ap.add_argument("--assignment-output", default="splits/celebdf_identity_assignments.csv")
    ap.add_argument("--excluded-output", default="splits/celebdf_identity_excluded.csv")
    ap.add_argument("--summary-output", default="splits/celebdf_identity_summary.json")
    args = ap.parse_args()
    if not 0.05 <= args.test_fraction <= 0.40:
        raise SystemExit("--test-fraction must be between 0.05 and 0.40")

    with open(args.input, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing = REQUIRED - set(fields)
        if missing:
            raise SystemExit(f"Input manifest missing columns: {sorted(missing)}")
        rows = list(reader)

    # Retain only source rows that were not part of the published official test.
    pool = [r for r in rows if r["split"] in {"train", "val"}]
    if not pool:
        raise SystemExit("No official train/val rows found in input manifest.")

    ids = sorted({token for r in pool
                  for token in ID_RE.findall(os.path.splitext(os.path.basename(r["abs_path"]))[0])})
    if len(ids) < 10:
        raise SystemExit(f"Only {len(ids)} identity IDs found; inspect filename convention before proceeding.")
    shuffled = ids[:]
    random.Random(args.seed).shuffle(shuffled)
    n_test = max(1, round(len(ids) * args.test_fraction))
    test_ids = set(shuffled[:n_test])
    train_ids = set(shuffled[n_test:])
    owner = {i: "test" for i in test_ids} | {i: "train" for i in train_ids}

    kept = []
    excluded = []
    reasons = Counter()
    for r in rows:
        if r["split"] == "test":
            excluded.append({**r, "exclusion_reason": "official_test_preserved_in_original_protocol"})
            reasons["official_test_preserved_in_original_protocol"] += 1
            continue
        tokens = list(dict.fromkeys(ID_RE.findall(os.path.splitext(os.path.basename(r["abs_path"]))[0])))
        label = r["label"].strip().lower()
        if not tokens:
            excluded.append({**r, "exclusion_reason": "no_identity_in_filename"})
            reasons["no_identity_in_filename"] += 1
            continue
        if label == "fake" and len(tokens) != 2:
            raise SystemExit(f"Expected two subject IDs in fake filename, found {tokens}: {r['abs_path']}")
        if label == "real" and len(tokens) != 1:
            raise SystemExit(f"Expected one subject ID in celebrity-real filename, found {tokens}: {r['abs_path']}")
        partitions = {owner[t] for t in tokens}
        if len(partitions) != 1:
            excluded.append({**r, "exclusion_reason": "fake_pair_crosses_identity_partitions"})
            reasons["fake_pair_crosses_identity_partitions"] += 1
            continue
        new_row = dict(r)
        new_row["split"] = next(iter(partitions))
        new_row["identity_ids"] = "|".join(tokens)
        kept.append(new_row)

    if not kept:
        raise SystemExit("No videos retained. Check the filename convention and chosen fraction.")
    counts = Counter((r["split"], r["label"].strip().lower()) for r in kept)
    # Assert disjoint identities by construction and after serialization.
    actual = defaultdict(set)
    seen_video_ids = defaultdict(set)
    for r in kept:
        for token in r["identity_ids"].split("|"):
            actual[r["split"]].add(token)
        if r["video_id"] in seen_video_ids[r["split"]]:
            raise SystemExit(f"Duplicate video_id within {r['split']}: {r['video_id']}")
        seen_video_ids[r["split"]].add(r["video_id"])
    overlap = actual["train"] & actual["test"]
    if overlap:
        raise SystemExit(f"Identity overlap remains: {sorted(overlap)}")

    output_fields = list(fields) + ["identity_ids"]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=output_fields)
        w.writeheader(); w.writerows(kept)
    with open(args.assignment_output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["identity_id", "split"])
        for identity in ids:
            w.writerow([identity, owner[identity]])
    excluded_fields = list(fields) + ["exclusion_reason"]
    with open(args.excluded_output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=excluded_fields)
        w.writeheader(); w.writerows(excluded)

    summary = {
        "protocol": "supplementary_identity_disjoint_celebdf",
        "input_manifest": args.input,
        "output_manifest": args.output,
        "official_test_rows_preserved_in_original_protocol_and_excluded_here": reasons["official_test_preserved_in_original_protocol"],
        "assignment_seed": args.seed,
        "test_identity_fraction_requested": args.test_fraction,
        "identities_total": len(ids),
        "n_identities_train": len(train_ids),
        "n_identities_test": len(test_ids),
        "identities_train": sorted(train_ids),
        "identities_test": sorted(test_ids),
        "videos_retained": len(kept),
        "video_counts_by_split_and_label": {f"{s}_{l}": n for (s, l), n in sorted(counts.items())},
        "videos_excluded_by_reason": dict(reasons),
        "train_test_identity_overlap_count": len(overlap),
        "protocol_notes": [
            "This is an additional experiment; the original official video-level test remains unchanged.",
            "Only official train/val videos are eligible for this additional experiment.",
            "Videos with no filename identity and fake pairs crossing identity partitions are excluded.",
            "All identity tokens in a fake filename must belong to the same partition.",
            "Verify the dataset's filename semantics and frame IDs before training."
        ],
    }
    with open(args.summary_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"identities_train", "identities_test"}}, indent=2, ensure_ascii=False))
    print("Identity overlap check: PASS (0 shared person IDs between alternate train/test).")


if __name__ == "__main__":
    main()
