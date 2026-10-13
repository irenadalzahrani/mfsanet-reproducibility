#!/bin/bash
# STEP 17: VREx experiment: quick debug -> 3 seeds of training -> evaluation -> branch-zeroing check. Background. Run AFTER s14 (finish) is done.
pgrep -f "[b]ash /workspace/run_vrex.sh" > /dev/null && { echo "ALREADY RUNNING: not starting a second copy"; exit 1; }
cd /workspace
[ -f results/ffpp_metrics_raw_per_seed.csv ] || echo "WARNING: results/ffpp_metrics_raw_per_seed.csv not found: run s14 first so the baseline exists"
nohup bash /workspace/run_vrex.sh > /workspace/logs/vrex.log 2>&1 &
echo "Started. Watch: tail -n 3 /workspace/logs/vrex.log   (finished when it ends with VREX_DONE; VREX_DEBUG_FAILED means stop and send me the log)"
