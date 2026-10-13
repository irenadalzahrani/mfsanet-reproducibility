#!/usr/bin/env python3
"""Adds the f3net_fad tag to eval_predictions.py (idempotent)."""
p = "/workspace/eval_predictions.py"; s = open(p).read()
if '"f3net_fad"' in s: print("already patched")
else:
    key = '"xception": lambda: XceptionBaseline(input_size=S),'
    assert key in s, "registry line not found"
    open(p, "w").write(s.replace(key, key + ' "f3net_fad": (lambda: __import__("models_f3net_fad").F3NetFAD(input_size=S)),', 1)); print("patched", p)
