import sys, os
sys.path.insert(0, "/workspace")
import torch
from train_common import train_one_seed
from models_xception import XceptionBaseline
from config import PATHS, TRAIN

device = "cuda" if torch.cuda.is_available() else "cpu"
tag = "xception"

for seed in [42, 43, 44]:
    if os.path.exists(f"{PATHS.ckpt_root}/{tag}/seed_{seed}/best.pt"):
        print(f"[skip] {tag} seed={seed}")
        continue
    torch.cuda.empty_cache()
    train_one_seed(tag, lambda: XceptionBaseline(input_size=TRAIN.image_size),
                   seed=seed, device=device, use_domain_adapt=False, use_freq_loss=False)

print("\n\n✅✅✅ XCEPTION BASELINE FINISHED ✅✅✅")
