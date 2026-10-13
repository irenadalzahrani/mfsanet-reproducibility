#!/usr/bin/env python3
"""Adds the mfsanet_irm tag to eval_predictions.py (idempotent)."""
p = "/workspace/eval_predictions.py"; s = open(p).read()
if '"mfsanet_irm"' in s: print("already patched")
else:
    key = '"mfsanet_A3": lambda: MFSANet(input_size=S, variant="full"),'
    assert key in s, "registry line not found"
    open(p, "w").write(s.replace(key, key + ' "mfsanet_irm": lambda: MFSANet(input_size=S, variant="full"),', 1)); print("patched", p)
