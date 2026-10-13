#!/bin/bash
cd /workspace
python3 patch_eval_irm.py
python3 train_irm.py --seeds 42 --quick-debug || { echo IRM_DEBUG_FAILED; exit 1; }
python3 train_irm.py --seeds 42 43 44
# evaluate ONLY the IRM model, keeping the full-run result files untouched (outputs get an _irm suffix)
for d in ffpp celebdf dfdc; do
  F="results/${d}_metrics_raw_per_seed.csv results/${d}_metrics_mean_std.csv predictions/${d}_per_video_scores_by_seed.csv predictions/${d}_seed_mean_video_scores.csv"
  for f in $F; do [ -f $f ] && cp $f $f.fullrun_backup; done
  python3 eval_predictions.py --dataset $d --models mfsanet_irm --seeds 42 43 44
  for f in $F; do mv $f ${f%.csv}_irm.csv; [ -f $f.fullrun_backup ] && mv $f.fullrun_backup $f; done
done
python3 zero_branch_check.py --models mfsanet_irm --datasets ffpp celebdf dfdc --seeds 42 43 44 --out branch_zeroing_irm.csv
echo IRM_DONE
