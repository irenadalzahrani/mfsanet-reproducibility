MFSA-Net: Reproducibility Package

Code, split manifests, per-video predictions and result tables for the associated paper. Trained model checkpoints are not included.

Requirements
Python 3.10
PyTorch 2.4.0
CUDA 12.1
timm (for the Xception baseline)
Structure
Models, training and evaluation (repository root)
models_mfsanet.py — MFSA-Net (all variants: Spatial-only, Spectral-only, Base, ZS, DA)
models_specxnet.py — SpecXNet baseline
models_defax.py — DeFaX baseline
models_xception.py — Xception baseline
train_common.py — unified training loop used for every model/variant
eval_common.py — full evaluation and branch-zeroing diagnostic
eval_predictions.py — metrics computed from saved per-video predictions
eval_identity_protocol.py — supplementary identity-disjoint Celeb-DF evaluation
robustness_common.py — JPEG/Gaussian-noise robustness sweep
run_da_baselines.py — trains the domain-adapted SpecXNet+DA and DeFaX+DA baselines (same mixed-domain corpus, recipe, and seeds as MFSA-Net-DA)
run_xception.py — example launcher script for the Xception baseline
config.py — all hyperparameters (TRAIN, PATHS)
datasets.py — dataset loading (DeepfakeFrameDataset)
extract_frames.py, check_raw_data.py — frame extraction and raw-data checks
Splits and verification
build_splits.py — generates the official video-disjoint split manifests
build_identity_splits.py — builds the supplementary identity-disjoint split
verify_splits.py — verifies split integrity (output saved in results/split_verification.txt)
splits/ — split manifests (video IDs) for FF++, CelebDF-v2, DFDC, and the identity-disjoint assignments
da_subset_videos.csv — videos selected for domain adaptation
Statistical analysis
bootstrap_analysis.py, bootstrap_official.py, bootstrap_identity.py — paired video-level bootstrap analyses
bootstrap_from_predictions.py — bootstrap computed directly from the released per-video predictions
bootstrap_paired_all.py — all paired comparisons reported in the paper (100,000 video-level resamples on seed-averaged scores)
benchmark_efficiency.py — parameter count, GMACs and latency
Extra experiments (addons/)

Branch-zeroing check, threshold calibration, V-REx and IRMv1 variants, the FAD branch of F3-Net, class-balancing ablations and the AUC-selected spectral-only variant. See the README_* files inside the folder.

Outputs
results/ — raw per-seed and mean/std test metrics (frame and video level, tau = 0.5 and 0.35) for every evaluated model, robustness sweeps, branch-zeroing results, threshold calibration and bootstrap results
predictions/ — per-video test scores per seed and seed-averaged, for all models and datasets
Datasets

FaceForensics++, CelebDF-v2, and DFDC must be obtained separately under their original license terms from their respective sources. This repository provides only the split manifests (which video IDs belong to train/val/test), not the raw video data itself.

Reproducing results
Download the three datasets per their original license terms.
Extract aligned face frames using MTCNN, following the split manifests in splits/.
Train a model with the unified loop in train_common.py, e.g.:
python
   from train_common import train_one_seed
   from models_mfsanet import MFSANet
   train_one_seed("mfsanet_v3", lambda: MFSANet(input_size=288, variant="full"),
                   seed=42, device="cuda", use_domain_adapt=True, use_freq_loss=True)
Evaluate with eval_common.run_full_evaluation(...) for test-set metrics (accuracy, AUC, and real-class F1, precision, recall at tau = 0.5 and tau = 0.35).
Run the robustness sweep with robustness_common.run_robustness_sweep(...).

All reported results average over three random seeds (42, 43, 44).

Conventions
Class labels: 1 = real, 0 = fake. The model's softmax output at index 1 is the real-class probability.
A video is labelled real if its mean real-class probability is >= tau. Two operating points are reported: tau = 0.5 and tau = 0.35. The value 0.35 is a fixed operating point whose original selection procedure was not recorded; a post-hoc calibration on the FF++ validation split is released in results/tau_calibration.csv.
Precision, recall and F1 use the real class as the positive class; AUC is unaffected by this choice.
Video-level scores are the mean of frame-level probabilities over the 20 frames sampled per video.
License

Code released for research reproducibility purposes accompanying the associated paper.
