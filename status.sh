#!/bin/bash
# Prints the state of everything. Run it any time:  bash /workspace/status.sh
echo "=== GPU / software ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "no GPU visible"
python3 - << 'PY' 2>/dev/null || echo "python packages NOT installed (run: bash /workspace/s01_setup.sh)"
import torch, timm
print("torch", torch.__version__, "| timm", timm.__version__, "| cuda available:", torch.cuda.is_available())
PY
echo "=== Kaggle token ==="; ls ~/.kaggle/kaggle.json 2>/dev/null || echo "missing (upload kaggle.json to /workspace and run: bash /workspace/s01_setup.sh)"
echo "=== raw data (videos) ==="
for d in ffpp celebdf dfdc; do
  n=$(find -L /workspace/data/raw/$d -name "*.mp4" 2>/dev/null | wc -l); s=$(du -shL /workspace/data/raw/$d 2>/dev/null | cut -f1)
  echo "$d: $n mp4 files, ${s:-0}"
done
echo "(expected mp4 counts: ffpp >= 6000, celebdf 6529, dfdc >= 6603)"
echo "=== frames (jpg) ==="
for d in ffpp celebdf dfdc celebdf_identity; do echo "$d: $(find /workspace/data/frames/$d -name '*.jpg' 2>/dev/null | wc -l) jpg"; done
echo "=== checkpoints (expected 39 at the very end) ==="; echo "$(ls /workspace/runs/*/seed_*/best.pt 2>/dev/null | wc -l) best.pt files"
echo "=== running jobs ==="; ps aux | grep -E "python3 (extract_frames|run_official|run_identity|eval_pred|run_robust)|wget|kaggle datasets|gdown" | grep -v grep | awk '{print $11, $12, $13, $14, $15}' | head -12
echo "=== last log lines ==="; for f in $(ls -t /workspace/logs/*.log 2>/dev/null | head -4); do echo "--- $f"; tail -n 2 "$f" | cut -c1-200; done
