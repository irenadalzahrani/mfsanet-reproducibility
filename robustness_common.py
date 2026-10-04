import csv, io, os
import numpy as np, torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from config import PATHS, ROBUST, TRAIN
from datasets import DeepfakeFrameDataset, IMAGENET_MEAN, IMAGENET_STD

def jpeg_degrade(img, q):
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=q); buf.seek(0)
    return Image.open(buf).convert("RGB")

def gaussian_degrade(img, sigma):
    arr = np.asarray(img).astype(np.float32)/255.0
    arr = np.clip(arr + np.random.normal(0,sigma,arr.shape).astype(np.float32),0,1)*255.0
    return Image.fromarray(arr.astype(np.uint8))

class RobustnessDataset(DeepfakeFrameDataset):
    def __init__(self, dataset, split, image_size, degrade_fn):
        super().__init__(dataset, split, image_size)
        self.degrade_fn = degrade_fn
        self.rc = transforms.Compose([transforms.Resize(int(image_size*256/224)), transforms.CenterCrop(image_size)])
        self.tt = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    def __getitem__(self, idx):
        path, label, vid = self.samples[idx]
        img = self.rc(Image.open(path).convert("RGB"))
        if self.degrade_fn: img = self.degrade_fn(img)
        return self.tt(img), label, 0, vid

@torch.no_grad()
def run_condition(model, dataset, split, degrade_fn, device):
    ds = RobustnessDataset(dataset, split, TRAIN.image_size, degrade_fn)
    loader = DataLoader(ds, batch_size=TRAIN.batch_size_per_step, shuffle=False, num_workers=4)
    probs, labels = [], []
    model.eval()
    for images, lbl, _f, _v in loader:
        p = torch.softmax(model(images.to(device)), dim=1)[:,1].cpu().numpy()
        probs.extend(p.tolist()); labels.extend(lbl.numpy().tolist())
    probs, labels = np.array(probs), np.array(labels)
    preds = (probs >= TRAIN.threshold_fixed).astype(int)
    auc = roc_auc_score(labels, probs) if len(set(labels.tolist()))>1 else float("nan")
    return auc, (preds==labels).mean()

def run_robustness_sweep(model_tag, build_model_fn, seeds, dataset, device):
    os.makedirs(PATHS.results_root, exist_ok=True)
    conditions = [("clean", None)]
    conditions += [(f"jpeg_q{q}", lambda img,q=q: jpeg_degrade(img,q)) for q in ROBUST.jpeg_qualities]
    conditions += [(f"gauss_s{s}", lambda img,s=s: gaussian_degrade(img,s)) for s in ROBUST.gaussian_sigmas]
    rows = [["model","seed","condition","auc","acc"]]
    for seed in seeds:
        ckpt_path = os.path.join(PATHS.ckpt_root, model_tag, f"seed_{seed}", "best.pt")
        if not os.path.exists(ckpt_path): continue
        model = build_model_fn().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
        for name, fn in conditions:
            auc, acc = run_condition(model, dataset, "test", fn, device)
            print(f"[{model_tag} seed={seed}] {name}: AUC={auc:.4f} ACC={acc:.4f}")
            rows.append([model_tag, seed, name, f"{auc:.4f}", f"{acc:.4f}"])
    out = os.path.join(PATHS.results_root, f"{model_tag}_robustness_sweep.csv")
    csv.writer(open(out,"w",newline="")).writerows(rows)
    print(f"[wrote] {out}")
