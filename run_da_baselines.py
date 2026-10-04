import sys, os
sys.path.insert(0, "/workspace")
import torch
from train_common import train_one_seed
from models_specxnet import SpecXNet
from models_defax import DeFaX
from config import PATHS, TRAIN

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

jobs_da = {
    "specxnet_da": (lambda: SpecXNet(input_size=TRAIN.image_size), True, False),
    "defax_da":    (lambda: DeFaX(input_size=TRAIN.image_size),    True, False),
}

for tag, (build_fn, da, freq) in jobs_da.items():
    for seed in [42, 43, 44]:
        ckpt_path = f"{PATHS.ckpt_root}/{tag}/seed_{seed}/best.pt"
        if os.path.exists(ckpt_path):
            print(f"\n[skip] {tag} seed={seed} already has a checkpoint — skipping.")
            continue
        torch.cuda.empty_cache()
        print(f"\n{'='*60}\nSTARTING {tag} seed={seed}\n{'='*60}")
        train_one_seed(tag, build_fn, seed=seed, device=device,
                        use_domain_adapt=da, use_freq_loss=freq)

print("\n\n✅✅✅ DA BASELINES TRAINING FINISHED ✅✅✅")
