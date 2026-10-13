#!/bin/bash
cd /workspace
if [ ! -f /workspace/DeepfakeBench/training/detectors/f3net_detector.py ]; then
  git clone --depth 1 https://github.com/SCLBD/DeepfakeBench /workspace/DeepfakeBench || { echo F3NET_CLONE_FAILED; exit 1; }
fi
echo "DeepfakeBench commit used: $(git -C /workspace/DeepfakeBench rev-parse HEAD 2>/dev/null)" | tee /workspace/logs/f3net_source_commit.txt
python3 patch_eval_f3net.py
python3 models_f3net_fad.py --selftest || { echo F3NET_SELFTEST_FAILED; exit 1; }
python3 run_f3net.py --seeds 42 --quick-debug || { echo F3NET_DEBUG_FAILED; exit 1; }
python3 run_f3net.py --seeds 42 43 44
echo F3NET_DONE
