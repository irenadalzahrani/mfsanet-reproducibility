#!/bin/bash
# STEP 14: evaluation + robustness sweep + statistics (after all training is done)
cd /workspace
grep -h DONE logs/pod1.log logs/pod2.log logs/pod3.log 2>/dev/null | grep POD || echo "WARNING: not all pods report DONE yet"
nohup bash /workspace/finish.sh > /workspace/logs/finish.log 2>&1 &
echo "Started. Finished when the last line of /workspace/logs/finish.log is FINISH_DONE:  tail -n 3 /workspace/logs/finish.log"
