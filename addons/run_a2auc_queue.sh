#!/bin/bash
cd /workspace
python3 patch_eval_a2auc.py
python3 run_a2_aucsel.py --seeds 42 --quick-debug || { echo A2AUC_DEBUG_FAILED; exit 1; }
python3 run_a2_aucsel.py --seeds 42 43 44
echo A2AUC_DONE
