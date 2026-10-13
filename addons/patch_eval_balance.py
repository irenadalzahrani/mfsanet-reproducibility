#!/usr/bin/env python3
"""Adds the balance-ablation tags to eval_predictions.py (idempotent)."""
p = "/workspace/eval_predictions.py"; s = open(p).read()
if '"mfsanet_A3_nosampler"' in s: print("already patched")
else:
    key = '"mfsanet_A3": lambda: MFSANet(input_size=S, variant="full"),'
    assert key in s, "registry line not found"
    add = ' "mfsanet_A3_nosampler": lambda: MFSANet(input_size=S, variant="full"), "mfsanet_A3_nofw": lambda: MFSANet(input_size=S, variant="full"), "mfsanet_A3_nobalance": lambda: MFSANet(input_size=S, variant="full"),'
    open(p, "w").write(s.replace(key, key + add, 1)); print("patched", p)
