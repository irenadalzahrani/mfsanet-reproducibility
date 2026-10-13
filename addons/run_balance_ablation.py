#!/usr/bin/env python3
"""Class-balancing ablation of the Base model (mfsanet_A3 = full MFSA-Net without Lfreq), same recipe / seeds as everywhere else.
Variants:  nosampler = plain shuffling instead of the class-balancing WeightedRandomSampler (fake-class loss weight kept)
           nofw      = fake-class loss weight 1.0 instead of 2.0 (weighted sampler kept)
           nobalance = both removed
Implemented by replacing two functions INSIDE THIS PROCESS ONLY (train_common.py is not modified), so the running experiments are unaffected.
    python3 run_balance_ablation.py --variant nosampler --seeds 42 43 44 [--quick-debug]     checkpoints: /workspace/runs/mfsanet_A3_<variant>/seed_<s>/best.pt"""
import argparse, functools, os, sys
sys.path.insert(0, "/workspace")
import torch
from torch.utils.data import RandomSampler
import train_common as tc
from config import PATHS, TRAIN
from models_mfsanet import MFSANet
ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True, choices=["nosampler", "nofw", "nobalance"])
ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44]); ap.add_argument("--quick-debug", action="store_true")
a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
if a.variant in ("nosampler", "nobalance"):
    tc.make_weighted_sampler = lambda ds: RandomSampler(ds)                                   # same number of samples per epoch, no class re-weighting
if a.variant in ("nofw", "nobalance"):
    tc.class_weighted_ce = functools.partial(tc.class_weighted_ce, fake_w=1.0)                # equal class weights in the loss (label smoothing unchanged)
tag = f"mfsanet_A3_{a.variant}"
print(f"variant={a.variant}  sampler replaced: {a.variant in ('nosampler', 'nobalance')}  fake-class loss weight: {1.0 if a.variant in ('nofw', 'nobalance') else TRAIN.fake_class_weight}", flush=True)
for s in a.seeds:
    ck = os.path.join(PATHS.ckpt_root, tag, f"seed_{s}", "best.pt")
    if os.path.exists(ck) and not a.quick_debug: print(f"[skip] {tag} seed={s}: checkpoint exists", flush=True); continue
    torch.cuda.empty_cache(); print(f"\n{'=' * 60}\nSTART {tag} seed={s}\n{'=' * 60}", flush=True)
    tc.train_one_seed(tag + ("_debug" if a.quick_debug else ""), lambda: MFSANet(input_size=TRAIN.image_size, variant="full"), seed=s, device=dev,
                      use_domain_adapt=False, use_freq_loss=False, quick_debug=a.quick_debug)
print("FINISHED", tag, a.seeds, flush=True)
