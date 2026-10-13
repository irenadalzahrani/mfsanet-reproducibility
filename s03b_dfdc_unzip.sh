#!/bin/bash
# STEP 3b: unzip DFDC parts and fix the folder layout expected by the split files
cd /workspace/data/raw/dfdc
if pgrep -x wget > /dev/null; then echo "wget is still downloading. Wait, then run this again."; exit 1; fi
for n in 01 02 03 04; do
  [ -f $n.zip ] || { echo "missing $n.zip"; continue; }
  mkdir -p dfdc_train_part_$n
  unzip -q -o $n.zip -d dfdc_train_part_$n && rm $n.zip
  p=$(echo $n | sed 's/^0//'); d=/workspace/data/raw/dfdc/dfdc_train_part_$n
  if ls $d/*.mp4 > /dev/null 2>&1; then mkdir -p $d/dfdc_train_part_$p; mv $d/*.mp4 $d/metadata.json $d/dfdc_train_part_$p/ 2>/dev/null; fi
  echo "part $n: $(find -L $d -name '*.mp4' | wc -l) videos in $(ls $d | head -1)"
done
echo "Expected folder names: dfdc_train_part_01/dfdc_train_part_1 ... dfdc_train_part_04/dfdc_train_part_4"
