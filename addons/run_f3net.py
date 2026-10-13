#!/usr/bin/env python3
"""Train the DeepfakeBench F3-Net (FAD) baseline with the SAME protocol as every other zero-shot model (FF++ only; same optimiser, losses, augmentation, seeds).
    python3 run_f3net.py --seeds 42 43 44 [--quick-debug]      checkpoints: /workspace/runs/f3net_fad/seed_<s>/best.pt"""
import argparse, os, sys
sys.path.insert(0, "/workspace")
import torch
from config import PATHS, TRAIN
from train_common import train_one_seed
from models_f3net_fad import F3NetFAD
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44]); ap.add_argument("--quick-debug", action="store_true")
a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
for s in a.seeds:
    ck = os.path.join(PATHS.ckpt_root, "f3net_fad", f"seed_{s}", "best.pt")
    if os.path.exists(ck) and not a.quick_debug: print(f"[skip] f3net_fad seed={s}: checkpoint exists", flush=True); continue
    torch.cuda.empty_cache(); print(f"\n{'=' * 60}\nSTART f3net_fad seed={s}\n{'=' * 60}", flush=True)
    train_one_seed("f3net_fad" + ("_debug" if a.quick_debug else ""), lambda: F3NetFAD(input_size=TRAIN.image_size), seed=s, device=dev,
                   use_domain_adapt=False, use_freq_loss=False, quick_debug=a.quick_debug)
print("FINISHED f3net_fad", a.seeds, flush=True)
