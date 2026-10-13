#!/bin/bash
cd /workspace
python3 patch_eval_registry.py
python3 train_vrex.py --seeds 42 --quick-debug || { echo VREX_DEBUG_FAILED; exit 1; }
python3 train_vrex.py --seeds 42 43 44
# evaluate ONLY the VREx model, keeping the full-run result files untouched (outputs get a _vrex suffix)
for d in ffpp celebdf dfdc; do
  F="results/${d}_metrics_raw_per_seed.csv results/${d}_metrics_mean_std.csv predictions/${d}_per_video_scores_by_seed.csv predictions/${d}_seed_mean_video_scores.csv"
  for f in $F; do [ -f $f ] && cp $f $f.fullrun_backup; done
  python3 eval_predictions.py --dataset $d --models mfsanet_vrex --seeds 42 43 44
  for f in $F; do mv $f ${f%.csv}_vrex.csv; [ -f $f.fullrun_backup ] && mv $f.fullrun_backup $f; done
done
python3 zero_branch_check.py --models mfsanet_vrex --datasets ffpp celebdf dfdc --seeds 42 43 44 --out branch_zeroing_vrex.csv
echo VREX_DONE
