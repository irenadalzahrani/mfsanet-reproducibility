#!/bin/bash
# Run ONCE (on one pod) after all frames are extracted and verified: builds the identity split, patches the code, creates celebdf_identity frames.
set -e
cd /workspace
python3 build_identity_splits.py --input data/splits/celebdf_splits.csv --output data/splits/celebdf_identity_splits.csv \
  --assignment-output data/splits/celebdf_identity_assignments.csv --excluded-output data/splits/celebdf_identity_excluded.csv \
  --summary-output data/splits/celebdf_identity_summary.json --seed 42 --test-fraction 0.20
python3 - << 'PY'
import json, sys
s = json.load(open("/workspace/data/splits/celebdf_identity_summary.json"))
exp = {"n_identities_train": 47, "n_identities_test": 12, "train_test_identity_overlap_count": 0}
cnt = s["video_counts_by_split_and_label"]; exc = s["videos_excluded_by_reason"]
ok = all(s[k] == v for k, v in exp.items()) and cnt == {"test_fake": 235, "test_real": 93, "train_fake": 3076, "train_real": 389} \
     and exc == {"no_identity_in_filename": 230, "fake_pair_crosses_identity_partitions": 1988, "official_test_preserved_in_original_protocol": 518}
print("identity split numbers match the expected values:", ok)
sys.exit(0 if ok else 1)
PY
if patch -p1 -N --dry-run < identity_protocol.patch > /dev/null 2>&1; then patch -p1 < identity_protocol.patch; else echo "patch already applied (or does not apply): check manually if unexpected"; fi
python3 reindex_identity_frames.py --manifest data/splits/celebdf_identity_splits.csv --frames-root data/frames \
  --source-dataset celebdf --output-dataset celebdf_identity --frames-per-video 20 --copy
echo PREPARE_IDENTITY_DONE
