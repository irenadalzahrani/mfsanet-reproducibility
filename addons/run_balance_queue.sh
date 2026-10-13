#!/bin/bash
V="$1"
cd /workspace
python3 patch_eval_balance.py
python3 run_balance_ablation.py --variant $V --seeds 42 --quick-debug || { echo BALANCE_${V}_DEBUG_FAILED; exit 1; }
python3 run_balance_ablation.py --variant $V --seeds 42 43 44
echo BALANCE_${V}_DONE
