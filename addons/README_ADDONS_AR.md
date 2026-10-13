# الإضافات (ارفعي هذا الملف مرة وحدة)
```bash
cd /workspace && unzip -o mfsanet_addons_all.zip > /dev/null && cp -r addon/* /workspace/ && chmod +x /workspace/*.sh && bash /workspace/apply_guards.sh && python3 /workspace/fix_done.py && python3 /workspace/patch_eval_registry.py
```
محتويات: train_vrex.py, zero_branch_check.py, calibrate_tau.py, compare_vrex.py, scan_frames.py, fix_done.py, apply_guards.sh, start_pod.sh, run_extra.sh, run_vrex.sh, s16_extra_analysis.sh, s17_vrex.sh, s18_vrex_compare.sh
