#!/usr/bin/env python3
"""MFSA-Net trained with a VREx penalty (Krueger et al., 2021) over the five FF++ manipulation methods as environments.
Zero-shot protocol is unchanged: training data = FF++ train only. A separate file, so the running pipeline (train_common.py) is untouched.
    python3 train_vrex.py --seeds 42 43 44 [--beta-max 10] [--lfreq] [--quick-debug]
Checkpoints: /workspace/runs/mfsanet_vrex[_lfreq]/seed_<s>/best.pt   (evaluate with eval_predictions.py after adding the tag; see README)"""
import argparse, json, os, random, sys, time
sys.path.insert(0, "/workspace")
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from config import PATHS, TRAIN
from datasets import DeepfakeFrameDataset, make_weighted_sampler
from train_common import set_seed, class_weighted_ce, mixup, evaluate_split
from models_mfsanet import MFSANet

METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]


def env_ids(video_ids):
    """-1 for real videos ('real_000'), 0..4 for the manipulation method that prefixes a fake video id ('Deepfakes_000_003')."""
    out = []
    for v in video_ids:
        p = str(v).split("_")[0]
        out.append(METHODS.index(p) if p in METHODS else -1)
    return torch.tensor(out)


def vrex_penalty(logits, labels, envs, device):
    """Variance of the per-environment risks. Environment e = fake samples of method e + all real samples of the batch,
    so the penalty measures how unevenly the model fits the different manipulation methods. Returns (penalty, n_envs)."""
    real = envs == -1
    if real.sum() == 0:
        return logits.new_zeros(()), 0
    risks = []
    for e in range(len(METHODS)):
        m = (envs == e) | real
        if (envs == e).sum() > 0:
            risks.append(class_weighted_ce(logits[m.to(device)], labels[m.to(device)], device))
    if len(risks) < 2:
        return logits.new_zeros(()), len(risks)
    return torch.stack(risks).var(unbiased=False), len(risks)


def train(seed, device, beta_max, ramp_epochs, use_lfreq, quick):
    name = "mfsanet_vrex" + ("_lfreq" if use_lfreq else "") + ("_debug" if quick else "")
    set_seed(seed)
    run_dir = os.path.join(PATHS.ckpt_root, name, f"seed_{seed}"); os.makedirs(run_dir, exist_ok=True)
    train_ds = DeepfakeFrameDataset("ffpp", "train", TRAIN.image_size); val_ds = DeepfakeFrameDataset("ffpp", "val", TRAIN.image_size)
    bs = TRAIN.batch_size_per_step
    tl = DataLoader(train_ds, batch_size=bs, sampler=make_weighted_sampler(train_ds), num_workers=4, pin_memory=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    model = MFSANet(input_size=TRAIN.image_size, variant="full").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN.base_lr, weight_decay=TRAIN.weight_decay)
    max_epochs = 2 if quick else TRAIN.max_epochs

    def lr_lambda(ep):
        if ep < TRAIN.warmup_epochs: return (ep + 1) / TRAIN.warmup_epochs
        prog = (ep - TRAIN.warmup_epochs) / max(1, (max_epochs - TRAIN.warmup_epochs))
        return max(TRAIN.eta_min / TRAIN.base_lr, 0.5 * (1 + np.cos(np.pi * prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    best, pat, log = -1.0, 0, []
    best_path = os.path.join(run_dir, "best.pt")
    for epoch in range(max_epochs):
        model.train(); opt.zero_grad(); run_loss = run_pen = 0.0; n_pen = 0; t0 = time.time()
        beta = beta_max * min(1.0, max(0.0, (epoch - TRAIN.warmup_epochs) / max(1, ramp_epochs)))   # penalty annealing: 0 during warm-up, then linear ramp
        for step, (images, labels, flag, vids) in enumerate(tl):
            images, labels, flag = images.to(device), labels.to(device), flag.to(device)
            pen = images.new_zeros(())
            if random.random() < TRAIN.mixup_prob:
                images, ya, yb, lam = mixup(images, labels, TRAIN.mixup_alpha, device)
                logits = model(images)
                loss = lam * class_weighted_ce(logits, ya, device) + (1 - lam) * class_weighted_ce(logits, yb, device)
            else:
                logits = model(images)
                loss = class_weighted_ce(logits, labels, device)
                pen, n_env = vrex_penalty(logits, labels, env_ids(vids), device)   # not on mixup steps: mixed samples have no single environment
                loss = loss + beta * pen
                run_pen += float(pen.detach()); n_pen += 1
            if use_lfreq and flag.sum() > 0:
                m = flag.bool(); loss = loss + TRAIN.freq_lambda * TRAIN.freq_weight * nn.CrossEntropyLoss(reduction="none")(logits[m], labels[m]).mean()
            (loss / TRAIN.grad_accum_steps).backward(); run_loss += loss.item()
            if (step + 1) % TRAIN.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN.grad_clip_norm); opt.step(); opt.zero_grad()
            if quick and step > 10: break
        sched.step()
        vm = evaluate_split(model, vl, device, TRAIN.threshold_calibrated)
        score = TRAIN.selection_auc_weight * vm["auc"] + TRAIN.selection_f1_weight * vm["f1"]
        log.append({"epoch": epoch, "train_loss": run_loss / max(1, len(tl)), "beta": beta, "mean_penalty": run_pen / max(1, n_pen), **vm, "score": score, "seconds": time.time() - t0})
        print(f"[{name} seed={seed}] epoch {epoch}: loss={log[-1]['train_loss']:.4f} beta={beta:.2f} vrex_pen={log[-1]['mean_penalty']:.5f} val_auc={vm['auc']:.4f} val_f1={vm['f1']:.4f} score={score:.4f}", flush=True)
        if score > best:
            best, pat = score, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_metrics": vm, "model_name": name, "seed": seed}, best_path)
        else:
            pat += 1
            if pat >= TRAIN.early_stopping_patience: print(f"[{name} seed={seed}] early stop @ epoch {epoch}"); break
        json.dump(log, open(os.path.join(run_dir, "train_log.json"), "w"), indent=2)
    print(f"[{name} seed={seed}] DONE. best={best_path} score={best:.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--beta-max", type=float, default=10.0)
    ap.add_argument("--ramp-epochs", type=int, default=10)
    ap.add_argument("--lfreq", action="store_true", help="also keep the Lfreq term (compare against mfsanet_A4); default off (compare against mfsanet_A3)")
    ap.add_argument("--quick-debug", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for s in a.seeds:
        ck = os.path.join(PATHS.ckpt_root, "mfsanet_vrex" + ("_lfreq" if a.lfreq else ""), f"seed_{s}", "best.pt")
        if os.path.exists(ck) and not a.quick_debug: print("[skip] exists:", ck); continue
        train(s, dev, a.beta_max, a.ramp_epochs, a.lfreq, a.quick_debug)
    print("FINISHED", a.seeds)
