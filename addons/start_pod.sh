#!/bin/bash
# usage: bash /workspace/start_pod.sh 1|2|3   (one queue per pod; see the runbook for what each queue trains)
N="$1"; [ -z "$N" ] && { echo "usage: bash /workspace/start_pod.sh 1|2|3"; exit 1; }
pgrep -f "[b]ash /workspace/pod${N}.sh" > /dev/null && { echo "ALREADY RUNNING: pod${N}.sh is already running on this pod"; exit 1; }
pgrep -f "[b]ash /workspace/pod[123].sh" > /dev/null && echo "NOTE: another queue is already running on THIS pod; they will share the GPU and both slow down"
cd /workspace && nohup bash /workspace/pod${N}.sh > /dev/null 2>&1 &
sleep 1; echo "pod${N}.sh started. Log: /workspace/logs/pod${N}.log   (finished when the log contains POD${N}_DONE)"
