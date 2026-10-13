#!/bin/bash
# STEP 33 (one pod): IRM experiment: quick debug -> 3 seeds of training -> evaluation -> branch-zeroing check. Background.
pgrep -f "[b]ash /workspace/run_irm.sh" > /dev/null && { echo "ALREADY RUNNING: not starting a second copy"; exit 1; }
cd /workspace
nohup bash /workspace/run_irm.sh > /workspace/logs/irm.log 2>&1 &
echo "Started. Watch: tail -n 5 /workspace/logs/irm.log   (IRM_DEBUG_FAILED = stop and send me the log; IRM_DONE = finished)"
