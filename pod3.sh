#!/bin/bash
# POD 3 (~58 GPU-hours): MFSA-Net variants, official adaptations, identity-protocol adaptation of MFSA-Net and SpecXNet
cd /workspace
python3 run_official_models.py --tags mfsanet_A3 mfsanet_A4 specxnet_da mfsanet_v3 --seeds 42 43 44 >> logs/pod3.log 2>&1
python3 run_identity_da.py --models mfsa specxnet --seeds 42 43 44 >> logs/pod3.log 2>&1
echo POD3_DONE >> logs/pod3.log
