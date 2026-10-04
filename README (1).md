# MFSA-Net: Reproducibility Package

Code and split manifests for the associated paper.

## Requirements
- Python 3.10
- PyTorch 2.4.0
- CUDA 12.1
- timm (for the Xception baseline)

## Structure
- `models_mfsanet.py` — MFSA-Net (all variants: Spatial-only, Spectral-only, Base, ZS, DA)
- `models_specxnet.py` — SpecXNet baseline
- `models_defax.py` — DeFaX baseline
- `models_xception.py` — Xception baseline
- `train_common.py` — unified training loop used for every model/variant
- `eval_common.py` — full evaluation and branch-zeroing diagnostic
- `robustness_common.py` — JPEG/Gaussian-noise robustness sweep
- `build_splits.py` — generates the official video-disjoint split manifests
- `config.py` — all hyperparameters (TRAIN, PATHS)
- `datasets.py` — dataset loading (DeepfakeFrameDataset)
- `run_xception.py` — example launcher script for the Xception baseline
- `splits/` — video-disjoint split manifests (video IDs) for FF++, CelebDF-v2, DFDC

## Datasets
FaceForensics++, CelebDF-v2, and DFDC must be obtained separately under their
original license terms from their respective sources. This repository provides
only the split manifests (which video IDs belong to train/val/test), not the
raw video data itself.

## Reproducing results
1. Download the three datasets per their original license terms.
2. Extract aligned face frames using MTCNN, following the split manifests in `splits/`.
3. Train a model with the unified loop in `train_common.py`, e.g.:
```python
   from train_common import train_one_seed
   from models_mfsanet import MFSANet
   train_one_seed("mfsanet_v3", lambda: MFSANet(input_size=288, variant="full"),
                   seed=42, device="cuda", use_domain_adapt=True, use_freq_loss=True)
```
4. Evaluate with `eval_common.run_full_evaluation(...)` for test-set metrics
   (accuracy, AUC, and real-class F1, precision, recall at both τ=0.5 and the calibrated τ=0.35).
5. Run the robustness sweep with `robustness_common.run_robustness_sweep(...)`.

All reported results average over three random seeds (42, 43, 44).

## License
Code released for research reproducibility purposes accompanying the associated paper.

## Additional scripts
- `run_da_baselines.py` — trains the domain-adapted SpecXNet+DA and DeFaX+DA baselines (same mixed-domain corpus, recipe, and seeds as MFSA-Net-DA)
- `bootstrap_analysis.py` — paired bootstrap (10,000 video-level resamples, seed-averaged scores) of MFSA-Net-DA vs. SpecXNet+DA / DeFaX+DA
- `predictions/` — seed-averaged per-video test scores used by the bootstrap analysis

## Conventions
- Class labels: 1 = real, 0 = fake. The model's softmax output at index 1 is the real-class probability.
- A video is labelled real if its mean real-class probability is >= tau (tau = 0.5 fixed, tau = 0.35 calibrated).
- Precision, recall and F1 use the real class as the positive class; AUC is unaffected by this choice.
- Video-level scores are the mean of frame-level probabilities over the 20 frames sampled per video.

## Results
- `results/` — raw per-seed and mean/std test metrics (frame and video level, tau = 0.5 and 0.35) for every evaluated model.
