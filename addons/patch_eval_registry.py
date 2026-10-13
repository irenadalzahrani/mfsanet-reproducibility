#!/usr/bin/env python3
"""Adds the VREx tags to eval_predictions.py so that  eval_predictions.py --models mfsanet_vrex  works. Idempotent."""
p = "/workspace/eval_predictions.py"
s = open(p).read()
if "mfsanet_vrex" in s:
    print("already patched")
else:
    key = '"mfsanet_A3": lambda: MFSANet(input_size=S, variant="full"),'
    assert key in s, "registry line not found"
    s = s.replace(key, key + ' "mfsanet_vrex": lambda: MFSANet(input_size=S, variant="full"), "mfsanet_vrex_lfreq": lambda: MFSANet(input_size=S, variant="full"),', 1)
    open(p, "w").write(s); print("patched", p)
