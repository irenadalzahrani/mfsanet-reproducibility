#!/bin/bash
# STEP 13: all training on THIS pod, cheapest models first (you can stop at any time with useful results)
cd /workspace
nohup bash -c "bash /workspace/pod3.sh; bash /workspace/pod2.sh; bash /workspace/pod1.sh" > /dev/null 2>&1 &
echo "Training started in the background. Watch:  bash /workspace/status.sh"
echo "Finished when logs/pod1.log, pod2.log and pod3.log each contain POD?_DONE. If the pod stops, run this script again: finished models are skipped."
