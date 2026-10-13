"""Paired video-level bootstrap computed from the released prediction files (no GPU, no checkpoints needed).
Run from the repository root:  python3 bootstrap_from_predictions.py > bootstrap_results.txt
For each comparison it prints (i) the observed AUC difference on the seed-averaged scores (deterministic) and (ii) the bootstrap
percentile interval (100,000 resamples of test videos, generator seed 42). The Monte Carlo error of the interval endpoints is about
+-0.0005 at 10,000 resamples and well below 0.0003 at 100,000 resamples, so report intervals to three decimals.
Scores are per-video scores averaged over the three training seeds (42, 43, 44) of each model."""
import sys
import numpy as np
from scipy.stats import rankdata

N_BOOT, SEED = 100000, 42


def auc(y, s):
    r = rankdata(s)
    npos = y.sum()
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))


def boot(a, b, labels, chunk=5000):
    rng = np.random.default_rng(SEED)
    d = []
    for _ in range(N_BOOT // chunk):
        idx = rng.integers(0, len(labels), (chunk, len(labels)))
        for i in idx:
            y = labels[i]
            if y.min() == y.max():
                continue
            d.append(auc(y, a[i]) - auc(y, b[i]))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return d.mean(), lo, hi, 100 * (d > 0).mean()


files = {"SpecXNet+DA": ("predictions/bootstrap_data.npy", "specx"), "DeFaX+DA": ("predictions/bootstrap_defax.npy", "defax")}
print(f"Paired bootstrap, {N_BOOT} resamples of test videos, generator seed {SEED}; delta = AUC(MFSA-Net-DA) - AUC(baseline+DA)")
for name, (path, key) in files.items():
    res = np.load(path, allow_pickle=True).item()
    for ds in ("celebdf", "dfdc"):
        if sys.argv[1:] and f"{ds}/{name}" not in sys.argv[1:]:
            continue  # optional filter, e.g.: python3 bootstrap_from_predictions.py dfdc/DeFaX+DA
        L, m, b = res[ds]["labels"].astype(int), res[ds]["mfsa"], res[ds][key]
        mean, lo, hi, fav = boot(m, b, L)
        print(f"{ds:8s} vs {name:12s}: observed delta={auc(L, m) - auc(L, b):+.4f} | bootstrap mean={mean:+.4f} 95% CI=[{lo:+.4f}, {hi:+.4f}] "
              f"favoring MFSA-Net-DA={fav:.1f}% | seed-averaged-score AUC: MFSA-Net-DA={auc(L, m):.4f} {name}={auc(L, b):.4f} "
              f"| videos={len(L)} (real={int((L == 1).sum())}, fake={int((L == 0).sum())})")
