#!/usr/bin/env python3
"""Extract 20 aligned face crops per video, in the layout that datasets.py expects:
    /workspace/data/frames/<dataset>/<split>/<real|fake>/<video_id>__f<NN>.jpg
Settings come from config.PREP (face_size=288, bbox_margin_px=20, frames_per_video=20, fallback_full_frame=True).
The original extraction script was not saved, so this is a re-implementation: frames are evenly spaced over the video, the largest MTCNN
face is cropped with a square box (+margin) and resized to face_size, and JPEGs are written with quality 95. Re-extracted frames therefore
match the original pipeline in format but are not bit-identical.

Examples (one process per shard; restartable):
    python3 extract_frames.py --dataset ffpp --shard 0 --num-shards 3 &
    python3 extract_frames.py --dataset dfdc --splits train --dfdc-parts 1 2
    python3 extract_frames.py --dataset celebdf --limit 5          # quick test on 5 videos
"""
import argparse, csv, os, sys, time
sys.path.insert(0, "/workspace")
import cv2, numpy as np, torch
from facenet_pytorch import MTCNN
from config import PATHS, PREP

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=["ffpp", "celebdf", "dfdc"])
ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
ap.add_argument("--dfdc-parts", nargs="+", type=int, default=[1, 2, 3, 4])
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0, help="process only the first N videos of this shard (test)")
ap.add_argument("--max-side", type=int, default=960, help="detect on a copy whose longer side is at most this")
ap.add_argument("--jpeg-quality", type=int, default=95)
args = ap.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
mtcnn = MTCNN(keep_all=True, device=device, post_process=False)

rows = list(csv.DictReader(open(os.path.join(PATHS.splits_root, f"{args.dataset}_splits.csv"), newline="", encoding="utf-8-sig")))
rows = [r for r in rows if r["split"] in args.splits]
if args.dataset == "dfdc":
    keep = {f"dfdc_train_part_{p}" for p in args.dfdc_parts}
    rows = [r for r in rows if r["source"] in keep]
rows = rows[args.shard::args.num_shards]
if args.limit:
    rows = rows[: args.limit]
prog_path = os.path.join(PATHS.frames_root, f"_progress_{args.dataset}_{args.shard}of{args.num_shards}.txt")
os.makedirs(PATHS.frames_root, exist_ok=True)
done = set(open(prog_path).read().split()) if os.path.exists(prog_path) else set()
fail_path = os.path.join(PATHS.frames_root, f"_failures_{args.dataset}_{args.shard}of{args.num_shards}.txt")
print(f"dataset={args.dataset} shard={args.shard}/{args.num_shards} videos={len(rows)} already_done={len(done)} device={device}", flush=True)


def sample_frames(path, n_want):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 0
        while cap.grab():
            total += 1
        cap.release(); cap = cv2.VideoCapture(path)
    if total == 0:
        return []
    idxs = sorted(set(np.linspace(0, total - 1, num=min(n_want, total)).round().astype(int).tolist()))
    want, last, frames, i = set(idxs), idxs[-1], [], 0
    while i <= last:
        if not cap.grab():
            break
        if i in want:
            ok, fr = cap.retrieve()
            if ok:
                frames.append(fr)
        i += 1
    cap.release()
    return frames


def largest_box(boxes):
    if boxes is None or len(boxes) == 0:
        return None
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    return boxes[int(np.argmax(areas))]


def crop(frame, box, margin, size):
    h, w = frame.shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = max(box[2] - box[0], box[3] - box[1]) + 2 * margin
    x1, y1, x2, y2 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2)), int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    c = frame[y1:y2, x1:x2]
    if c.size == 0:
        return None
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA if min(c.shape[:2]) > size else cv2.INTER_LINEAR)


t0, n_ok, n_fail = time.time(), 0, 0
for k, r in enumerate(rows, 1):
    vid = r["video_id"]
    if vid in done:
        continue
    try:
        frames = sample_frames(r["abs_path"], PREP.frames_per_video)
        if not frames:
            raise RuntimeError("no frames decoded")
        h, w = frames[0].shape[:2]
        s = min(1.0, args.max_side / max(h, w))
        small = [cv2.cvtColor(cv2.resize(f, (int(w * s), int(h * s))) if s < 1 else f, cv2.COLOR_BGR2RGB) for f in frames]
        boxes_list, _ = mtcnn.detect(np.stack(small))
        out_dir = os.path.join(PATHS.frames_root, args.dataset, r["split"], r["label"])
        os.makedirs(out_dir, exist_ok=True)
        last_box = None
        for j, fr in enumerate(frames):
            b = largest_box(boxes_list[j]) if boxes_list is not None else None
            if b is not None:
                b = np.asarray(b, dtype=float) / s
                last_box = b
            elif last_box is not None:
                b = last_box
            img = crop(fr, b, PREP.bbox_margin_px, PREP.face_size) if b is not None else None
            if img is None:
                if not PREP.fallback_full_frame:
                    continue
                img = cv2.resize(fr, (PREP.face_size, PREP.face_size))
            cv2.imwrite(os.path.join(out_dir, f"{vid}__f{j:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        with open(prog_path, "a") as f:
            f.write(vid + "\n")
        n_ok += 1
    except Exception as e:  # keep going; failures are listed for a re-run
        n_fail += 1
        with open(fail_path, "a") as f:
            f.write(f"{vid}\t{r['abs_path']}\t{e}\n")
    if k % 50 == 0:
        el = time.time() - t0
        print(f"[{k}/{len(rows)}] ok={n_ok} fail={n_fail} elapsed={el/60:.1f} min  eta={(len(rows)-k)*el/max(k,1)/60:.1f} min", flush=True)
print(f"FINISHED shard {args.shard}: ok={n_ok} fail={n_fail} (failures listed in {fail_path})", flush=True)
