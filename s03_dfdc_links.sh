#!/bin/bash
# STEP 3: DFDC parts 01-04. You paste the four links (copied from the Kaggle Data page); downloads run in the background.
mkdir -p /workspace/data/raw/dfdc /workspace/logs
cd /workspace/data/raw/dfdc
for n in 01 02 03 04; do
  read -r -p "Paste the link of $n.zip, then press Enter: " U
  echo "check for $n.zip:"; wget --spider -S "$U" 2>&1 | grep -E "HTTP/|Content-Length" | head -3
  nohup wget -c -O "$n.zip" "$U" > /workspace/logs/dl_dfdc_$n.log 2>&1 &
done
echo; echo "4 downloads started (about 9.4 GB each). Watch:  ls -la /workspace/data/raw/dfdc"
echo "When every zip is about 9.4 GB and 'pgrep wget' prints nothing, run:  bash /workspace/s03b_dfdc_unzip.sh"
