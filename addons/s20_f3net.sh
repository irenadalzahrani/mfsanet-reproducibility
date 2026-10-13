#!/bin/bash
# STEP 20 (on its OWN pod): F3-Net (DeepfakeBench FAD implementation): clone -> self-test -> quick debug -> 3 seeds. Background.
pgrep -f "[b]ash /workspace/run_f3net_queue.sh" > /dev/null && { echo "ALREADY RUNNING: not starting a second copy"; exit 1; }
cd /workspace
nohup bash /workspace/run_f3net_queue.sh > /workspace/logs/f3net.log 2>&1 &
echo "Started. Watch: tail -n 6 /workspace/logs/f3net.log   (F3NET_CLONE_FAILED / F3NET_SELFTEST_FAILED / F3NET_DEBUG_FAILED = stop and send me the log; F3NET_DONE = finished)"
