#!/bin/bash
# STEP 11: evaluate the gate model and compare with the old run
cd /workspace
python3 eval_predictions.py --dataset ffpp --models specxnet --seeds 42 2>&1 | tail -4
python3 eval_predictions.py --dataset celebdf --models specxnet --seeds 42 2>&1 | tail -4
python3 show_auc.py ffpp celebdf
echo "ACCEPT if FF++ AUC is between 0.960 and 0.985 and CelebDF AUC between 0.74 and 0.88 (old run: 0.9728 and 0.7909). Otherwise STOP and send me this output."
