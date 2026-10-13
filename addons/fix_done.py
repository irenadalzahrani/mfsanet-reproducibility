p = "/workspace/done.py"; s = open(p).read()
old = 'print("\\nNEXT:"'
new = 'nxt = [x for x in nxt if not (running and (x.startswith("Remaining frames") or x.startswith("Frames not extracted")))]\nprint("\\nNEXT:"'
if 'x.startswith("Remaining frames")' in s.split("NEXT")[-2][-300:]: print("done.py: already patched")
else: open(p, "w").write(s.replace(old, new, 1)); print("done.py: patched")
