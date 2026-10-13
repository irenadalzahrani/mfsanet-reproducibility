#!/bin/bash
# STEP 30 (own pod): class-balancing ablation.  usage: bash /workspace/s30_balance.sh nosampler|nofw
V="$1"; [ "$V" = "nosampler" ] || [ "$V" = "nofw" ] || { echo "usage: bash /workspace/s30_balance.sh nosampler|nofw"; exit 1; }
pgrep -f "[b]ash /workspace/run_balance_queue.sh $V" > /dev/null && { echo "ALREADY RUNNING: not starting a second copy"; exit 1; }
cd /workspace
nohup bash /workspace/run_balance_queue.sh $V > /workspace/logs/balance_$V.log 2>&1 &
echo "Started. Watch: tail -n 5 /workspace/logs/balance_$V.log   (BALANCE_${V}_DEBUG_FAILED = stop and send me the log; BALANCE_${V}_DONE = finished)"
