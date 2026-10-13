#!/usr/bin/env python3
"""Paired video-level bootstrap for the official protocol, from the per-video scores written by eval_predictions.py
(predictions/<dataset>_seed_mean_video_scores.csv). Same method and output format as bootstrap_from_predictions.py.
    python3 bootstrap_official.py --datasets celebdf dfdc --comparisons mfsanet_v3:specxnet_da mfsanet_v3:defax_da"""
import argparse, csv, sys
import numpy as np
from scipy.stats import rankdata

ap = argparse.ArgumentParser()
ap.add_argument("--predictions-dir", default="/workspace/predictions")
ap.add_argument("--datasets", nargs="+", default=["celebdf", "dfdc"])
ap.add_argument("--comparisons", nargs="+", default=["mfsanet_v3:specxnet_da", "mfsanet_v3:defax_da"])
ap.add_argument("--n-bootstrap", type=int, default=100000)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()


def auc(y, s):
    r = rankdata(s); npos = y.sum()
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))


def boot(a, b, labels, chunk=5000):
    rng = np.random.default_rng(args.seed); d = []
    for _ in range(args.n_bootstrap // chunk):
        idx = rng.integers(0, len(labels), (chunk, len(labels)))
        for i in idx:
            y = labels[i]
            if y.min() == y.max():
                continue
            d.append(auc(y, a[i]) - auc(y, b[i]))
    d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
    return d.mean(), lo, hi, 100 * (d > 0).mean()


print(f"Paired bootstrap, {args.n_bootstrap} resamples of test videos, generator seed {args.seed}; delta = AUC(model_a) - AUC(model_b)")
for ds in args.datasets:
    data = {}
    for r in csv.DictReader(open(f"{args.predictions_dir}/{ds}_seed_mean_video_scores.csv", newline="", encoding="utf-8-sig")):
        data.setdefault(r["model"], {})[r["video_id"]] = (int(r["y_true"]), float(r["p_real_mean"]), int(r["n_seeds"]))
    for pair in args.comparisons:
        ma, mb = pair.split(":")
        if ma not in data or mb not in data:
            print(f"{ds:8s} {pair}: missing predictions", flush=True); continue
        vids = sorted(set(data[ma]) & set(data[mb]))
        assert set(data[ma]) == set(data[mb]), "models were scored on different videos"
        L = np.array([data[ma][v][0] for v in vids]); A = np.array([data[ma][v][1] for v in vids]); B = np.array([data[mb][v][1] for v in vids])
        assert all(data[ma][v][0] == data[mb][v][0] for v in vids), "label mismatch"
        mean, lo, hi, fav = boot(A, B, L)
        print(f"{ds:8s} {ma} vs {mb}: observed delta={auc(L, A) - auc(L, B):+.4f} | bootstrap mean={mean:+.4f} 95% CI=[{lo:+.4f}, {hi:+.4f}] favoring {ma}={fav:.1f}% "
              f"| seed-averaged-score AUC: {ma}={auc(L, A):.4f} {mb}={auc(L, B):.4f} | videos={len(L)} (real={int((L == 1).sum())}, fake={int((L == 0).sum())}) "
              f"seeds={sorted({data[ma][v][2] for v in vids})}", flush=True)
