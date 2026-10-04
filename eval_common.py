import collections, csv, os
import numpy as np, torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from config import PATHS, TRAIN
from datasets import DeepfakeFrameDataset

@torch.no_grad()
def frame_level_predictions(model, loader, device):
    model.eval(); probs, labels, vids = [], [], []
    for images, lbl, _freq, vid in loader:
        images = images.to(device)
        p = torch.softmax(model(images), dim=1)[:,1].cpu().numpy()
        probs.extend(p.tolist()); labels.extend(lbl.numpy().tolist()); vids.extend(list(vid))
    return np.array(probs), np.array(labels), np.array(vids)

def video_level_aggregate(probs, labels, vids):
    by_p, by_l = collections.defaultdict(list), {}
    for p,l,v in zip(probs, labels, vids): by_p[v].append(p); by_l[v]=l
    vs = sorted(by_p.keys())
    return np.array([np.mean(by_p[v]) for v in vs]), np.array([by_l[v] for v in vs])

def compute_metrics(probs, labels, thr):
    preds = (probs >= thr).astype(int)
    auc = roc_auc_score(labels, probs) if len(set(labels.tolist()))>1 else float("nan")
    return {"auc":auc, "acc":(preds==labels).mean(),
            "precision":precision_score(labels,preds,zero_division=0),
            "recall":recall_score(labels,preds,zero_division=0),
            "f1":f1_score(labels,preds,zero_division=0)}

def evaluate_checkpoint(build_model_fn, ckpt_path, dataset, split, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model_fn().to(device)
    model.load_state_dict(ckpt["model_state"])
    ds = DeepfakeFrameDataset(dataset, split, TRAIN.image_size)
    loader = DataLoader(ds, batch_size=TRAIN.batch_size_per_step, shuffle=False, num_workers=4)
    probs, labels, vids = frame_level_predictions(model, loader, device)
    vprobs, vlabels = video_level_aggregate(probs, labels, vids)
    out = {"n_frames": len(probs), "n_videos": len(vprobs)}
    for name, thr in (("fixed_0.5", TRAIN.threshold_fixed), ("calibrated_0.35", TRAIN.threshold_calibrated)):
        out[f"frame_{name}"] = compute_metrics(probs, labels, thr)
        out[f"video_{name}"] = compute_metrics(vprobs, vlabels, thr)
    return out

def mean_std(vals):
    a = np.array(vals, dtype=float); return float(np.nanmean(a)), float(np.nanstd(a))

def run_full_evaluation(model_tag, build_model_fn, seeds, datasets, device):
    os.makedirs(PATHS.results_root, exist_ok=True)
    rows, per_key = [], collections.defaultdict(list)
    for seed in seeds:
        ckpt = os.path.join(PATHS.ckpt_root, model_tag, f"seed_{seed}", "best.pt")
        if not os.path.exists(ckpt): print(f"[skip] no ckpt seed={seed}"); continue
        for ds in datasets:
            try:
                res = evaluate_checkpoint(build_model_fn, ckpt, ds, "test", device)
            except RuntimeError as e:
                print(f"[skip] {ds}: {e}"); continue
            print(f"[{model_tag} seed={seed} {ds}] {res}")
            for level in ("frame","video"):
                for thr in ("fixed_0.5","calibrated_0.35"):
                    for m,v in res[f"{level}_{thr}"].items():
                        per_key[(ds,level,thr,m)].append(v)
                        rows.append([model_tag,seed,ds,level,thr,m,v])
    csv.writer(open(os.path.join(PATHS.results_root, f"{model_tag}_raw_per_seed.csv"),"w",newline="")).writerows(
        [["model","seed","dataset","level","threshold","metric","value"]] + rows)
    with open(os.path.join(PATHS.results_root, f"{model_tag}_summary_mean_std.csv"),"w",newline="") as f:
        w = csv.writer(f); w.writerow(["model","dataset","level","threshold","metric","mean","std","n_seeds"])
        for (ds,level,thr,m), vals in sorted(per_key.items()):
            mean, std = mean_std(vals); w.writerow([model_tag,ds,level,thr,m,f"{mean:.4f}",f"{std:.4f}",len(vals)])
    print(f"\n=== {model_tag}: AUC (video-level, mean ± std) ===")
    for ds in datasets:
        key = (ds,"video","fixed_0.5","auc")
        if key in per_key:
            mean, std = mean_std(per_key[key]); print(f"  {ds:10s}: {mean:.4f} ± {std:.4f}")
