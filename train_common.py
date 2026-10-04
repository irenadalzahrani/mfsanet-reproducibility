import json, os, random, time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
from config import PATHS, TRAIN
from datasets import DeepfakeFrameDataset, MixedDomainDataset, make_weighted_sampler

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def class_weighted_ce(logits, labels, device, label_smoothing=TRAIN.label_smoothing, fake_w=TRAIN.fake_class_weight):
    w = torch.tensor([fake_w, 1.0], device=device)
    return nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)(logits, labels)

def mixup(x, y, alpha, device):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=device)
    return lam*x + (1-lam)*x[perm], y, y[perm], lam

@torch.no_grad()
def evaluate_split(model, loader, device, threshold):
    model.eval()
    probs, labels = [], []
    for images, lbl, _freq, _vid in loader:
        images = images.to(device)
        p = torch.softmax(model(images), dim=1)[:,1].cpu().numpy()
        probs.extend(p.tolist()); labels.extend(lbl.numpy().tolist())
    probs, labels = np.array(probs), np.array(labels)
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(labels, probs) if len(set(labels.tolist()))>1 else float("nan")
    return {"auc": float(auc), "f1": float(f1_score(labels, preds)), "acc": float((preds==labels).mean())}

def train_one_seed(model_name, build_model_fn, seed, device, use_domain_adapt=False,
                    use_freq_loss=False, quick_debug=False, max_epochs_override=None):
    set_seed(seed)
    run_dir = os.path.join(PATHS.ckpt_root, model_name, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    train_ds = MixedDomainDataset(TRAIN.image_size) if use_domain_adapt else DeepfakeFrameDataset("ffpp","train",TRAIN.image_size)
    val_ds = DeepfakeFrameDataset("ffpp","val",TRAIN.image_size)
    sampler = make_weighted_sampler(train_ds)
    bs = TRAIN.batch_size_per_step
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model_fn().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN.base_lr, weight_decay=TRAIN.weight_decay)
    max_epochs = max_epochs_override or (2 if quick_debug else TRAIN.max_epochs)

    def lr_lambda(epoch):
        if epoch < TRAIN.warmup_epochs: return (epoch+1)/TRAIN.warmup_epochs
        progress = (epoch-TRAIN.warmup_epochs)/max(1,(max_epochs-TRAIN.warmup_epochs))
        cos = 0.5*(1+np.cos(np.pi*progress))
        return max(TRAIN.eta_min/TRAIN.base_lr, cos)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_score, patience_ctr, log = -1.0, 0, []
    best_path = os.path.join(run_dir, "best.pt")

    for epoch in range(max_epochs):
        model.train(); optimizer.zero_grad(); running_loss = 0.0; t0 = time.time()
        for step, (images, labels, freq_flag, _vid) in enumerate(train_loader):
            images, labels, freq_flag = images.to(device), labels.to(device), freq_flag.to(device)
            if random.random() < TRAIN.mixup_prob:
                images, ya, yb, lam = mixup(images, labels, TRAIN.mixup_alpha, device)
                logits = model(images)
                base_loss = lam*class_weighted_ce(logits, ya, device) + (1-lam)*class_weighted_ce(logits, yb, device)
            else:
                logits = model(images)
                base_loss = class_weighted_ce(logits, labels, device)

            if use_freq_loss and freq_flag.sum() > 0:
                mask = freq_flag.bool()
                l_freq = nn.CrossEntropyLoss(reduction="none")(logits[mask], labels[mask]).mean()
                loss = base_loss + TRAIN.freq_lambda * TRAIN.freq_weight * l_freq
            else:
                loss = base_loss

            (loss/TRAIN.grad_accum_steps).backward()
            running_loss += loss.item()
            if (step+1) % TRAIN.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN.grad_clip_norm)
                optimizer.step(); optimizer.zero_grad()
            if quick_debug and step > 10: break

        scheduler.step()
        val_metrics = evaluate_split(model, val_loader, device, TRAIN.threshold_calibrated)
        score = TRAIN.selection_auc_weight*val_metrics["auc"] + TRAIN.selection_f1_weight*val_metrics["f1"]
        entry = {"epoch":epoch, "train_loss":running_loss/max(1,len(train_loader)), **val_metrics,
                 "score":score, "seconds":time.time()-t0}
        log.append(entry)
        print(f"[{model_name} seed={seed}] epoch {epoch}: loss={entry['train_loss']:.4f} "
              f"val_auc={val_metrics['auc']:.4f} val_f1={val_metrics['f1']:.4f} score={score:.4f}")

        if score > best_score:
            best_score, patience_ctr = score, 0
            torch.save({"model_state":model.state_dict(),"epoch":epoch,"val_metrics":val_metrics,
                        "model_name":model_name,"seed":seed}, best_path)
        else:
            patience_ctr += 1
            if patience_ctr >= TRAIN.early_stopping_patience:
                print(f"[{model_name} seed={seed}] early stop @ epoch {epoch}"); break
        json.dump(log, open(os.path.join(run_dir,"train_log.json"),"w"), indent=2)

    print(f"[{model_name} seed={seed}] DONE. best={best_path} score={best_score:.4f}")
    return best_path
