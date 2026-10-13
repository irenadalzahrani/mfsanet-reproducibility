#!/bin/bash
# stops s07 / s13 / s14 from starting a second copy of a job that is already running
cd /workspace
add_guard() { f="$1"; pat="$2"; if grep -q "ALREADY RUNNING" "$f"; then echo "$f: already guarded"; else sed -i "1a pgrep -f \"$pat\" > /dev/null \&\& { echo \"ALREADY RUNNING: not starting a second copy\"; exit 1; }" "$f"; echo "$f: guard added"; fi; }
add_guard s07_extract_all.sh "[b]ash /workspace/run_extract_all.sh"
add_guard s13_train_one_pod.sh "[b]ash /workspace/pod[123].sh"
add_guard s14_finish.sh "[b]ash /workspace/finish.sh"
