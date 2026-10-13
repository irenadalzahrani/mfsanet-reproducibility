#!/bin/bash
# STEP 1: environment + Kaggle token + folders
cd /workspace
chmod +x /workspace/*.sh
bash /workspace/pod_setup.sh
pip install -q kaggle gdown
if [ -f /workspace/kaggle.json ]; then mkdir -p ~/.kaggle && mv /workspace/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json && echo "Kaggle token installed"; fi
mkdir -p /workspace/data/raw /workspace/data/frames /workspace/runs /workspace/results /workspace/predictions /workspace/logs
echo; echo "######## STATE ########"; bash /workspace/status.sh
