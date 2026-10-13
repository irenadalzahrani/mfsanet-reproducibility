#!/bin/bash
# STEP 10: quality gate = train SpecXNet (seed 42) in the background and compare with the old run
cd /workspace
nohup python3 run_official_models.py --tags specxnet --seeds 42 > logs/gate_specxnet_42.log 2>&1 &
echo "Started. Watch:  tail -n 3 /workspace/logs/gate_specxnet_42.log   and the GPU:  nvidia-smi"
echo "When the log prints 'DONE. best=...', run:  bash /workspace/s11_gate_eval.sh"
