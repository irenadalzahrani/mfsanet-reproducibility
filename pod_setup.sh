#!/bin/bash
# Run once on a fresh pod (from /workspace):   bash pod_setup.sh
set -e
cd /workspace
echo "=== machine ==="; python --version; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "=== installing packages (facenet-pytorch WITHOUT dependencies, so it does not replace the pod's torch) ==="
pip install --no-deps facenet-pytorch
pip install timm opencv-python-headless scikit-learn scipy pandas tqdm requests
mkdir -p /workspace/data/raw /workspace/data/frames /workspace/runs /workspace/results /workspace/predictions /workspace/logs
python - << 'PY'
import torch, timm, cv2, sklearn, facenet_pytorch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| timm", timm.__version__, "| opencv", cv2.__version__)
print("GPU available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
pip freeze > /workspace/requirements_pod.txt
echo "environment recorded in /workspace/requirements_pod.txt (use it to correct the software versions stated in the paper)"
