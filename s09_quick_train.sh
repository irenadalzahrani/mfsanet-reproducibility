#!/bin/bash
# STEP 9: 2 short epochs for two models, to catch errors early (writes *_debug checkpoints only)
cd /workspace
python3 run_official_models.py --tags specxnet mfsanet_A4 --seeds 42 --quick-debug 2>&1 | tail -30
