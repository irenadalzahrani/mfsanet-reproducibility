#!/bin/bash
# STEP 4: CelebDF-v2 from the link you received from the dataset authors
read -r -p "Paste the CelebDF download link, then press Enter: " U
mkdir -p /workspace/data/raw/celebdf_download && cd /workspace/data/raw/celebdf_download
if echo "$U" | grep -q "drive.google"; then
  if echo "$U" | grep -q "/folders/"; then gdown --folder "$U"; else gdown --fuzzy "$U"; fi
else
  wget -c --content-disposition "$U"
fi
for z in $(find . -maxdepth 3 -name "*.zip"); do unzip -q -o "$z"; done
mkdir -p /workspace/data/raw/celebdf
for d in Celeb-real Celeb-synthesis YouTube-real; do
  P=$(find /workspace/data/raw/celebdf_download -maxdepth 5 -type d -iname "$d" | head -1)
  if [ -n "$P" ]; then ln -sfn "$P" /workspace/data/raw/celebdf/$d; echo "linked $d -> $P"; else echo "NOT FOUND: $d"; fi
done
for d in Celeb-real Celeb-synthesis YouTube-real; do echo "$d: $(ls /workspace/data/raw/celebdf/$d 2>/dev/null | wc -l) files"; done
echo "Expected: Celeb-real 590, Celeb-synthesis 5639, YouTube-real 300"
