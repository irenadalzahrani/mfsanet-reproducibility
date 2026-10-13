#!/bin/bash
# STEP 6: face extraction on 5 videos per dataset (look at the images afterwards)
cd /workspace
python3 extract_frames.py --dataset ffpp --limit 5 2>&1 | tail -2
python3 extract_frames.py --dataset celebdf --limit 5 2>&1 | tail -2
python3 extract_frames.py --dataset dfdc --limit 5 2>&1 | tail -2
echo "--- sample files:"; find /workspace/data/frames -name "*.jpg" | head -6
echo "Open two of these images in JupyterLab (double click): the face must be visible and centered."
