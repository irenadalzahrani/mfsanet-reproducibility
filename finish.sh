#!/bin/bash
# Run after POD1_DONE, POD2_DONE and POD3_DONE appear in the logs: evaluation, robustness sweep, statistics, split verification.
cd /workspace
for ds in ffpp celebdf dfdc; do
  python3 eval_predictions.py --dataset $ds --models specxnet defax xception mfsanet_A1 mfsanet_A2 mfsanet_A3 mfsanet_A4 mfsanet_v3 specxnet_da defax_da --seeds 42 43 44
done
python3 eval_predictions.py --dataset celebdf_identity --models mfsanet_A4 specxnet defax mfsanet_identity_da specxnet_identity_da defax_identity_da --seeds 42 43 44
python3 run_robustness.py
python3 bootstrap_official.py --datasets celebdf dfdc | tee results/official_bootstrap.txt
python3 bootstrap_identity.py --n-bootstrap 100000 --seed 42
python3 verify_splits.py > results/split_verification.txt
python3 benchmark_efficiency.py --device cuda --iters 30 > results/benchmark_gpu.txt 2>&1
echo FINISH_DONE
