#!/bin/bash
# POD 2 (~71 GPU-hours): DeFaX official adaptation, SpecXNet, Xception, spatial-only, spectral-only
cd /workspace
python3 run_official_models.py --tags defax_da specxnet xception mfsanet_A1 mfsanet_A2 --seeds 42 43 44 >> logs/pod2.log 2>&1
echo POD2_DONE >> logs/pod2.log
