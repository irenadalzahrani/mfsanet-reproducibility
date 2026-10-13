#!/bin/bash
# STEP 32 (any free pod): spectral-only model with AUC-based checkpoint selection, 3 seeds. Background.
pgrep -f "[b]ash /workspace/run_a2auc_queue.sh" > /dev/null && { echo "ALREADY RUNNING: not starting a second copy"; exit 1; }
cd /workspace
nohup bash /workspace/run_a2auc_queue.sh > /workspace/logs/a2auc.log 2>&1 &
echo "Started. Watch: tail -n 5 /workspace/logs/a2auc.log   (A2AUC_DEBUG_FAILED = stop and send me the log; A2AUC_DONE = finished)"
