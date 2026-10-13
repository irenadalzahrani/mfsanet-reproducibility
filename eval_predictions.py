#!/usr/bin/env python3
"""Evaluate any trained checkpoint (official-protocol or identity-protocol) on the TEST split of --dataset (ffpp, celebdf, dfdc, celebdf_identity).

Writes per-seed frame/video metrics and per-video probabilities. The identity
scores are saved per seed and averaged across seeds for reproducible paired
bootstrap analysis.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
sys.path.insert(0, "/workspace")
import numpy as np
import torch
from torch.utils.data import DataLoader
from config import PATHS, TRAIN
from datasets import DeepfakeFrameDataset
from models_mfsanet import MFSANet
from models_specxnet import SpecXNet
from models_defax import DeFaX
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def build_registry():
    S = TRAIN.image_size
    from models_xception import XceptionBaseline
    return {
        "specxnet": lambda: SpecXNet(input_size=S, pretrained=False), "defax": lambda: DeFaX(input_size=S, pretrained=False),
        "xception": lambda: XceptionBaseline(input_size=S),
        "mfsanet_A1": lambda: MFSANet(input_size=S, variant="spatial_only"), "mfsanet_A2": lambda: MFSANet(input_size=S, variant="spectral_only"),
        "mfsanet_A3": lambda: MFSANet(input_size=S, variant="full"), "mfsanet_A4": lambda: MFSANet(input_size=S, variant="full"),
        "mfsanet_v3": lambda: MFSANet(input_size=S, variant="full"),
        "specxnet_da": lambda: SpecXNet(input_size=S, pretrained=False), "defax_da": lambda: DeFaX(input_size=S, pretrained=False),
        "mfsanet_identity_da": lambda: MFSANet(input_size=S, variant="full"),
        "specxnet_identity_da": lambda: SpecXNet(input_size=S, pretrained=False), "defax_identity_da": lambda: DeFaX(input_size=S, pretrained=False),
    }


def compute_metrics(scores, labels, threshold):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred = (scores >= threshold).astype(int)
    auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan")
    return {
        "auc": auc,
        "acc": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(labels, pred, pos_label=1, zero_division=0)),
    }


def predict(model, dataset, device):
    loader = DataLoader(dataset, batch_size=TRAIN.batch_size_per_step, shuffle=False, num_workers=4, pin_memory=True)
    frame_scores, frame_labels, frame_ids = [], [], []
    model.eval()
    with torch.no_grad():
        for images, labels, _freq, video_ids in loader:
            logits = model(images.to(device))
            p_real = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            frame_scores.extend(p_real.tolist())
            frame_labels.extend(labels.numpy().astype(int).tolist())
            frame_ids.extend(list(video_ids))
    grouped = defaultdict(list)
    group_labels = defaultdict(set)
    for score, label, video_id in zip(frame_scores, frame_labels, frame_ids):
        grouped[str(video_id)].append(float(score))
        group_labels[str(video_id)].add(int(label))
    inconsistent = [vid for vid, labs in group_labels.items() if len(labs) != 1]
    if inconsistent:
        raise RuntimeError(f"Inconsistent frame labels for {len(inconsistent)} videos: {inconsistent[:10]}")
    video_ids_sorted = sorted(grouped)
    video_scores = [float(np.mean(grouped[v])) for v in video_ids_sorted]
    video_labels = [next(iter(group_labels[v])) for v in video_ids_sorted]
    return (frame_scores, frame_labels, frame_ids,
            video_scores, video_labels, video_ids_sorted,
            {v: len(grouped[v]) for v in video_ids_sorted})


def main():
    registry = build_registry()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", choices=list(registry), default=list(registry))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--dataset", required=True, choices=["ffpp", "celebdf", "dfdc", "celebdf_identity"])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.35])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "dataset:", args.dataset)
    dataset = DeepfakeFrameDataset(args.dataset, "test", TRAIN.image_size)
    if len(dataset) == 0:
        raise SystemExit("Identity test dataset has no frames.")

    raw_metrics = []
    video_prediction_rows = []
    skipped = []
    for tag in args.models:
        builder = registry[tag]
        for seed in args.seeds:
            ckpt_path = os.path.join(PATHS.ckpt_root, tag, f"seed_{seed}", "best.pt")
            if not os.path.isfile(ckpt_path):
                skipped.append((tag, seed, ckpt_path))
                print(f"[skip] missing checkpoint: {ckpt_path}")
                continue
            checkpoint = torch.load(ckpt_path, map_location=device)
            model = builder().to(device)
            model.load_state_dict(checkpoint["model_state"])
            (frame_scores, frame_labels, frame_ids, video_scores, video_labels,
             video_ids, frames_per_video) = predict(model, dataset, device)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            for level, scores, labels in (
                ("frame", frame_scores, frame_labels),
                ("video", video_scores, video_labels),
            ):
                for threshold in args.thresholds:
                    for metric, value in compute_metrics(scores, labels, threshold).items():
                        raw_metrics.append({
                            "model": tag, "seed": seed, "dataset": args.dataset,
                            "level": level, "threshold": f"{threshold:.2f}",
                            "metric": metric, "value": value,
                        })
            for video_id, label, score in zip(video_ids, video_labels, video_scores):
                video_prediction_rows.append({
                    "model": tag, "seed": seed, "dataset": args.dataset,
                    "video_id": video_id, "y_true": label,
                    "p_real": score, "n_frames": frames_per_video[video_id],
                })
            print(f"[done] {tag} seed={seed}: frames={len(frame_scores)} videos={len(video_scores)}")

    if not raw_metrics:
        raise SystemExit("No model checkpoints were evaluated.")
    os.makedirs(PATHS.results_root, exist_ok=True)
    pred_dir = os.path.join(os.path.dirname(PATHS.results_root.rstrip("/")), "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    raw_path = os.path.join(PATHS.results_root, f"{args.dataset}_metrics_raw_per_seed.csv")
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        cols = ["model", "seed", "dataset", "level", "threshold", "metric", "value"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(raw_metrics)

    groups = defaultdict(list)
    for row in raw_metrics:
        key = (row["model"], row["dataset"], row["level"], row["threshold"], row["metric"])
        groups[key].append(float(row["value"]))
    summary_path = os.path.join(PATHS.results_root, f"{args.dataset}_metrics_mean_std.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        cols = ["model", "dataset", "level", "threshold", "metric", "mean", "std", "n_seeds"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for key, values in sorted(groups.items()):
            arr = np.asarray(values, dtype=float)
            w.writerow(dict(zip(cols, [*key, float(np.nanmean(arr)), float(np.nanstd(arr)), int(len(arr))])))

    per_seed_path = os.path.join(pred_dir, f"{args.dataset}_per_video_scores_by_seed.csv")
    with open(per_seed_path, "w", newline="", encoding="utf-8") as f:
        cols = ["model", "seed", "dataset", "video_id", "y_true", "p_real", "n_frames"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(video_prediction_rows)

    # Average probabilities across seeds per video; retain labels and seed counts for bootstrap.
    by_video = defaultdict(list)
    for row in video_prediction_rows:
        by_video[(row["model"], row["dataset"], row["video_id"])].append(row)
    mean_path = os.path.join(pred_dir, f"{args.dataset}_seed_mean_video_scores.csv")
    with open(mean_path, "w", newline="", encoding="utf-8") as f:
        cols = ["model", "dataset", "video_id", "y_true", "p_real_mean", "n_seeds", "n_frames"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for (tag, ds, vid), rows in sorted(by_video.items()):
            labels = {int(r["y_true"]) for r in rows}
            if len(labels) != 1:
                raise SystemExit(f"Label mismatch across seeds for {tag}/{vid}")
            w.writerow({"model": tag, "dataset": ds, "video_id": vid, "y_true": next(iter(labels)),
                        "p_real_mean": float(np.mean([float(r["p_real"]) for r in rows])),
                        "n_seeds": len(rows), "n_frames": rows[0]["n_frames"]})

    print("\nWrote:")
    print(" ", raw_path)
    print(" ", summary_path)
    print(" ", per_seed_path)
    print(" ", mean_path)
    if skipped:
        print("\nMissing checkpoints:", skipped)


if __name__ == "__main__":
    main()
