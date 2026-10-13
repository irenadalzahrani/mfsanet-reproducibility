#!/bin/bash
# STEP 31: after the extra experiments are finished, make finish.sh evaluate them together with all the other models (idempotent)
cd /workspace
python3 - << 'PY'
import glob, re
tags = ["f3net_fad", "mfsanet_A3_nosampler", "mfsanet_A3_nofw"]
done = set()
for f in glob.glob("/workspace/logs/*.log"):
    for l in open(f, errors="ignore"):
        m = re.match(r"\[(\S+) seed=(\d+)\] DONE", l)
        if m: done.add((m.group(1), int(m.group(2))))
ok = [t for t in tags if all((t, s) in done for s in (42, 43, 44))]
lines = open("/workspace/finish.sh").read().splitlines(); added = set()
for i, line in enumerate(lines):
    if "--dataset $ds --models" in line:
        for t in ok:
            if f" {t} " not in line + " ":
                line = line.replace(" --seeds 42 43 44", f" {t} --seeds 42 43 44"); added.add(t)
        lines[i] = line
open("/workspace/finish.sh", "w").write("\n".join(lines) + "\n")
print("experiments finished (3 seeds):", ok, "| not finished yet:", [t for t in tags if t not in ok], "| added to finish.sh now:", sorted(added))
print([l for l in lines if "--dataset $ds --models" in l][0])
PY
