import sys, numpy as np, torch
sys.path.insert(0, ".")
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from config import PATHS, TRAIN
from datasets import DeepfakeFrameDataset
from eval_common import video_level_aggregate
from models_mfsanet import MFSANet
from models_specxnet import SpecXNet
from models_defax import DeFaX

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def video_scores(build_fn, ckpt, dataset):
    model = build_fn().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"]); model.eval()
    ds = DeepfakeFrameDataset(dataset, "test", TRAIN.image_size)
    P, L, V = [], [], []
    for x, y, _f, v in DataLoader(ds, batch_size=16, shuffle=False, num_workers=0):
        P += torch.softmax(model(x.to(device)), 1)[:, 1].cpu().tolist(); L += y.tolist(); V += list(v)
    return video_level_aggregate(np.array(P), np.array(L), np.array(V))

baselines = {"specxnet_da": lambda: SpecXNet(input_size=TRAIN.image_size),
             "defax_da": lambda: DeFaX(input_size=TRAIN.image_size)}
mfsa = lambda: MFSANet(input_size=TRAIN.image_size, variant="full")
rng = np.random.default_rng(42)

for tag, build in baselines.items():
    for dataset in ["celebdf", "dfdc"]:
        sb, sm = [], []
        for seed in [42, 43, 44]:
            b, labels = video_scores(build, f"{PATHS.ckpt_root}/{tag}/seed_{seed}/best.pt", dataset)
            m, _ = video_scores(mfsa, f"{PATHS.ckpt_root}/mfsanet_v3/seed_{seed}/best.pt", dataset)
            sb.append(b); sm.append(m)
        b, m = np.mean(sb, 0), np.mean(sm, 0)
        deltas = []
        for _ in range(10000):
            i = rng.integers(0, len(labels), len(labels))
            if len(np.unique(labels[i])) < 2: continue
            deltas.append(roc_auc_score(labels[i], m[i]) - roc_auc_score(labels[i], b[i]))
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        print(f"{dataset} MFSA-Net-DA vs {tag}: dAUC={np.mean(deltas):+.4f} CI=[{lo:+.4f},{hi:+.4f}] favoring MFSA={np.mean(np.array(deltas)>0)*100:.1f}%")
