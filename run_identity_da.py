#!/usr/bin/env python3
"""Train the domain-adapted models for the separate CelebDF identity protocol.

Uses the existing FF++ train set, a fixed number of `celebdf_identity/train`
videos equal to the official protocol's 30% CelebDF video budget, and the
existing 20% DFDC train subset. Checkpoints use new tags and do not overwrite
official-protocol checkpoints.
"""
import argparse
import csv
import os
import sys
sys.path.insert(0, "/workspace")
import torch
from config import PATHS, TRAIN, SPLITS
from train_common import train_one_seed
from models_mfsanet import MFSANet
from models_specxnet import SpecXNet
from models_defax import DeFaX


def count_videos(path, wanted_split):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = csv.DictReader(f)
        return len({r["video_id"] for r in rows if r["split"] == wanted_split})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", choices=["mfsa", "specxnet", "defax"], default=["mfsa", "specxnet", "defax"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("CelebDF dataset:", "celebdf_identity")
    print("Early stopping/selection remains on FF++ validation, as in the main protocol.")
    official_train_n = count_videos("/workspace/data/splits/celebdf_splits.csv", "train")
    identity_train_n = count_videos("/workspace/data/splits/celebdf_identity_splits.csv", "train")
    target_video_budget = int(official_train_n * SPLITS.celebdf_train_frac_for_da)
    if target_video_budget > identity_train_n:
        print(f"WARNING: identity train pool has only {identity_train_n} videos; "
              f"cannot match official target budget {target_video_budget}.")
        target_video_budget = identity_train_n
    print(f"Matched CelebDF target video budget: {target_video_budget} "
          f"(official train={official_train_n}, identity train={identity_train_n}).")
    jobs = {
        "mfsa": ("mfsanet_identity_da", lambda: MFSANet(input_size=TRAIN.image_size, variant="full"), True),
        "specxnet": ("specxnet_identity_da", lambda: SpecXNet(input_size=TRAIN.image_size), False),
        "defax": ("defax_identity_da", lambda: DeFaX(input_size=TRAIN.image_size), False),
    }
    for key in args.models:
        tag, build_fn, freq_loss = jobs[key]
        for seed in args.seeds:
            ckpt = os.path.join(PATHS.ckpt_root, tag, f"seed_{seed}", "best.pt")
            if os.path.exists(ckpt):
                print(f"[skip] {tag} seed={seed} already has checkpoint: {ckpt}")
                continue
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"\n{'=' * 60}\nSTART {tag} seed={seed}\n{'=' * 60}")
            train_one_seed(
                tag,
                build_fn,
                seed=seed,
                device=device,
                use_domain_adapt=True,
                use_freq_loss=freq_loss,
                target_celebdf_dataset="celebdf_identity",
                target_celebdf_videos=target_video_budget,
            )
    print("Identity-protocol DA training finished.")


if __name__ == "__main__":
    main()
