#!/bin/bash
# STEP 8: frame counts + split checks
cd /workspace
cat /workspace/logs/extract_done.txt 2>/dev/null || echo "WARNING: extraction has not finished yet"
python3 verify_frames.py
echo "--- failures (should be empty or very short):"; cat data/frames/_failures_*.txt 2>/dev/null | head
python3 verify_splits.py | head -12
