#!/usr/bin/env python3
"""Hard-link existing CelebDF train/val frames into a separate identity split.

This does not overwrite the original official-video split. Frame names are
expected to start with the manifest video_id followed by '__f', matching
DeepfakeFrameDataset's filename parser.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="splits/celebdf_identity_splits.csv")
    ap.add_argument("--frames-root", default="/workspace/data/frames")
    ap.add_argument("--source-dataset", default="celebdf")
    ap.add_argument("--output-dataset", default="celebdf_identity")
    ap.add_argument("--frames-per-video", type=int, default=20)
    ap.add_argument("--copy", action="store_true", help="Copy files instead of hard-linking them")
    args = ap.parse_args()

    with open(args.manifest, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"video_id", "split", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise SystemExit(f"Manifest must contain columns {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise SystemExit("Identity manifest is empty.")

    root = Path(args.frames_root)
    output = root / args.output_dataset
    if output.exists() and any(output.rglob("*.jpg")):
        raise SystemExit(f"Output already contains JPGs: {output}. Move/delete it explicitly before rebuilding.")

    wanted = {(r["video_id"], r["label"], r["split"]) for r in rows}
    if len(wanted) != len(rows):
        raise SystemExit("Duplicate (video_id,label,split) entries in identity manifest.")
    by_video = {r["video_id"]: r for r in rows}
    discovered = defaultdict(list)
    # The identity protocol uses only original official train/val pool videos.
    for old_split in ("train", "val"):
        for label in ("real", "fake"):
            source_dir = root / args.source_dataset / old_split / label
            if not source_dir.is_dir():
                continue
            for path in source_dir.glob("*.jpg"):
                video_id = path.name.split("__f", 1)[0]
                if video_id not in by_video:
                    continue
                row = by_video[video_id]
                if row["label"] != label:
                    raise SystemExit(f"Label mismatch for frame {path.name}: manifest={row['label']} dir={label}")
                discovered[video_id].append(path)

    missing = sorted(set(by_video) - set(discovered))
    if missing:
        raise SystemExit(f"No existing train/val frames found for {len(missing)} identity-split videos. Examples: {missing[:10]}")
    too_many = {vid: len(paths) for vid, paths in discovered.items() if len(paths) > args.frames_per_video}
    if too_many:
        raise SystemExit(f"More than {args.frames_per_video} frames/video for {len(too_many)} videos. Examples: {list(too_many.items())[:10]}")
    fewer = {vid: len(paths) for vid, paths in discovered.items() if len(paths) < args.frames_per_video}
    if fewer:  # a few short videos legitimately yield fewer than 20 frames; allowed, but reported
        print(f"NOTE: {len(fewer)} videos have fewer than {args.frames_per_video} frames (kept). Examples: {list(fewer.items())[:5]}")

    method_counts = Counter()
    for video_id, paths in discovered.items():
        r = by_video[video_id]
        dest_dir = output / r["split"] / r["label"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in paths:
            dst = dest_dir / src.name
            if dst.exists():
                continue
            if args.copy:
                shutil.copy2(src, dst)
                method_counts["copied"] += 1
            else:
                try:
                    os.link(src, dst)
                    method_counts["hardlinked"] += 1
                except OSError:
                    shutil.copy2(src, dst)
                    method_counts["copied_fallback"] += 1

    # Verify all expected frames arrived in the correct split/label directory.
    bad = []
    for video_id, paths in discovered.items():
        r = by_video[video_id]
        dest_dir = output / r["split"] / r["label"]
        found = list(dest_dir.glob(video_id + "__f*.jpg"))
        if len(found) != len(paths):
            bad.append((video_id, len(found), r["split"], r["label"]))
    if bad:
        raise SystemExit(f"Post-copy verification failed for {len(bad)} videos: {bad[:10]}")

    summary = {
        "input_manifest": args.manifest,
        "source_frames": str(root / args.source_dataset),
        "output_frames": str(output),
        "videos": len(by_video),
        "frames_per_video_expected": args.frames_per_video,
        "frames_linked_or_copied": sum(method_counts.values()),
        "file_operations": dict(method_counts),
        "video_counts": dict(Counter(r["split"] for r in rows)),
        "label_counts": {f"{s}_{l}": sum(1 for r in rows if r["split"] == s and r["label"] == l)
                         for s in sorted({r["split"] for r in rows}) for l in ("real", "fake")},
        "note": "Original /celebdf frame directories were read only; the new dataset lives under /celebdf_identity."
    }
    summary_path = output / "frame_reindex_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
