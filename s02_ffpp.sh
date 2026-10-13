#!/bin/bash
# STEP 2: FF++ (Kaggle mirror xdxd003/ff-c23), runs in the background
mkdir -p /workspace/data/raw/ffpp /workspace/logs
echo "--- files that Kaggle lists for this dataset:"; kaggle datasets files xdxd003/ff-c23 2>&1 | head -12
nohup kaggle datasets download -d xdxd003/ff-c23 -p /workspace/data/raw/ffpp --unzip > /workspace/logs/dl_ffpp.log 2>&1 &
echo; echo "FF++ download started in the background."
echo "Watch it with:  du -sh /workspace/data/raw/ffpp   and   tail -n 3 /workspace/logs/dl_ffpp.log"
