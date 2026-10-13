#!/usr/bin/env python3
p = "/workspace/eval_predictions.py"; s = open(p).read()
if '"mfsanet_A2_aucsel"' in s: print("already patched")
else:
    key = '"mfsanet_A2": lambda: MFSANet(input_size=S, variant="spectral_only"),'
    assert key in s, "registry line not found"
    open(p, "w").write(s.replace(key, key + ' "mfsanet_A2_aucsel": lambda: MFSANet(input_size=S, variant="spectral_only"),', 1)); print("patched", p)
