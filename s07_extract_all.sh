#!/bin/bash
# STEP 7: full face extraction in the background (3 parallel workers per dataset)
cat > /workspace/run_extract_all.sh << 'INNER'
#!/bin/bash
cd /workspace
for ds in ffpp celebdf dfdc; do
  for i in 0 1 2; do python3 extract_frames.py --dataset $ds --shard $i --num-shards 3 > logs/ext_${ds}_$i.log 2>&1 & done
  wait
done
echo EXTRACT_ALL_DONE > logs/extract_done.txt
INNER
rm -f /workspace/logs/extract_done.txt
nohup bash /workspace/run_extract_all.sh > /workspace/logs/run_extract_all.log 2>&1 &
echo "Extraction started in the background. Watch:  bash /workspace/status.sh   (finished when logs/extract_done.txt exists)"
