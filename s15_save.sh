#!/bin/bash
# STEP 15: pack the results (download outputs.zip from JupyterLab, then Stop the pod)
cd /workspace && zip -r -q outputs.zip results predictions logs data/splits requirements_pod.txt && ls -la /workspace/outputs.zip
