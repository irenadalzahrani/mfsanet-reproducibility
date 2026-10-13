#!/bin/bash
cd /workspace
python3 patch_eval_registry.py
python3 zero_branch_check.py --models mfsanet_A3 mfsanet_A4 --datasets ffpp celebdf dfdc --seeds 42 43 44
python3 calibrate_tau.py --models specxnet defax xception mfsanet_A3 mfsanet_A4 specxnet_da defax_da mfsanet_v3 --seeds 42 43 44
echo EXTRA_DONE
