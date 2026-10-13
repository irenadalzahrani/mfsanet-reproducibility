#!/usr/bin/env python3
"""Train official-protocol models by tag, exactly as the original runs (same train_common.train_one_seed, same seeds).
    python3 run_official_models.py --tags specxnet defax mfsanet_A4 --seeds 42 43 44
    python3 run_official_models.py --tags specxnet --seeds 42 --quick-debug      # 2 short epochs, to test the pipeline
Tags: specxnet, defax, xception (FF++ only); mfsanet_A3 (Base), mfsanet_A4 (ZS = Base + Lfreq); mfsanet_v3, specxnet_da, defax_da (domain adapted).
mfsanet_A1 / mfsanet_A2 (spatial-only / spectral-only) assume the frequency loss was OFF: confirm before relying on them (not needed for the identity experiment).
Existing checkpoints are skipped, so the script can be restarted safely."""
import argparse, os, sys
sys.path.insert(0, "/workspace")
import torch
from config import PATHS, TRAIN
from train_common import train_one_seed
from models_mfsanet import MFSANet
from models_specxnet import SpecXNet
from models_defax import DeFaX
from models_xception import XceptionBaseline

S = TRAIN.image_size
JOBS = {  # tag: (builder, use_domain_adapt, use_freq_loss)
    "specxnet": (lambda: SpecXNet(input_size=S), False, False), "defax": (lambda: DeFaX(input_size=S), False, False),
    "xception": (lambda: XceptionBaseline(input_size=S), False, False),
    "mfsanet_A1": (lambda: MFSANet(input_size=S, variant="spatial_only"), False, False),
    "mfsanet_A2": (lambda: MFSANet(input_size=S, variant="spectral_only"), False, False),
    "mfsanet_A3": (lambda: MFSANet(input_size=S, variant="full"), False, False),
    "mfsanet_A4": (lambda: MFSANet(input_size=S, variant="full"), False, True),
    "mfsanet_v3": (lambda: MFSANet(input_size=S, variant="full"), True, True),
    "specxnet_da": (lambda: SpecXNet(input_size=S), True, False), "defax_da": (lambda: DeFaX(input_size=S), True, False),
}
ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", required=True, choices=list(JOBS))
ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
ap.add_argument("--quick-debug", action="store_true")
args = ap.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| tags:", args.tags, "| seeds:", args.seeds, flush=True)
for tag in args.tags:
    build, da, freq = JOBS[tag]
    for seed in args.seeds:
        ckpt = os.path.join(PATHS.ckpt_root, tag, f"seed_{seed}", "best.pt")
        if os.path.exists(ckpt) and not args.quick_debug:
            print(f"[skip] {tag} seed={seed}: checkpoint exists", flush=True)
            continue
        torch.cuda.empty_cache()
        print(f"\n{'=' * 60}\nSTART {tag} seed={seed} (domain_adapt={da}, freq_loss={freq})\n{'=' * 60}", flush=True)
        train_one_seed(tag + ("_debug" if args.quick_debug else ""), build, seed=seed, device=device, use_domain_adapt=da, use_freq_loss=freq,
                       quick_debug=args.quick_debug)
print("\nFINISHED", args.tags, args.seeds, flush=True)
