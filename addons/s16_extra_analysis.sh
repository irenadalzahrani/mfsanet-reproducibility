#!/bin/bash
# STEP 16: branch-zeroing check (per seed) + validation-based threshold calibration. Needs the trained checkpoints. Runs in the background.
pgrep -f "[b]ash /workspace/run_extra.sh" > /dev/null && { echo "ALREADY RUNNING: not starting a second copy"; exit 1; }
cd /workspace
echo "checkpoints found: $(ls runs/*/seed_*/best.pt 2>/dev/null | wc -l)"
nohup bash /workspace/run_extra.sh > /workspace/logs/extra.log 2>&1 &
echo "Started. Watch: tail -n 3 /workspace/logs/extra.log   (finished when it ends with EXTRA_DONE)"
