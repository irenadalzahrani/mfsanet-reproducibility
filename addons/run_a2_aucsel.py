#!/usr/bin/env python3
"""Spectral-only model (mfsanet_A2) with the checkpoint selected by validation AUC alone (selection weights 1.0 AUC + 0.0 F1) instead of the shared
0.6*AUC + 0.4*F1 at tau=0.35. Reason: for a near-chance model the shared rule picked epoch 0 (val AUC ~0.53) although val AUC rose to ~0.59-0.60 by epoch 14.
Everything else (recipe, seeds, early stopping patience 15, 60-epoch cap) is identical. The change is made inside this process only.
    python3 run_a2_aucsel.py --seeds 42 43 44 [--quick-debug]      checkpoints: /workspace/runs/mfsanet_A2_aucsel/seed_<s>/best.pt"""
import argparse, os, sys
sys.path.insert(0, "/workspace")
import torch
import train_common as tc
from config import PATHS, TRAIN
from models_mfsanet import MFSANet
TRAIN.selection_auc_weight = 1.0; TRAIN.selection_f1_weight = 0.0
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44]); ap.add_argument("--quick-debug", action="store_true")
a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; tag = "mfsanet_A2_aucsel"
print(f"selection weights: AUC {TRAIN.selection_auc_weight}, F1 {TRAIN.selection_f1_weight}", flush=True)
for s in a.seeds:
    ck = os.path.join(PATHS.ckpt_root, tag, f"seed_{s}", "best.pt")
    if os.path.exists(ck) and not a.quick_debug: print(f"[skip] {tag} seed={s}: checkpoint exists", flush=True); continue
    torch.cuda.empty_cache(); print(f"\n{'=' * 60}\nSTART {tag} seed={s}\n{'=' * 60}", flush=True)
    tc.train_one_seed(tag + ("_debug" if a.quick_debug else ""), lambda: MFSANet(input_size=TRAIN.image_size, variant="spectral_only"), seed=s, device=dev,
                      use_domain_adapt=False, use_freq_loss=False, quick_debug=a.quick_debug)
print("FINISHED", tag, a.seeds, flush=True)
