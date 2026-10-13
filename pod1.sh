#!/bin/bash
# POD 1 (~82 GPU-hours): DeFaX zero-shot, then DeFaX identity-protocol adaptation
cd /workspace
python3 run_official_models.py --tags defax --seeds 42 43 44 >> logs/pod1.log 2>&1
python3 run_identity_da.py --models defax --seeds 42 43 44 >> logs/pod1.log 2>&1
echo POD1_DONE >> logs/pod1.log
