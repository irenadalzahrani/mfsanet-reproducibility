# معايرة العتبة τ على بيانات التحقق (بعد التدريب)
```bash
cd /workspace && unzip -o mfsanet_tau_addon.zip > /dev/null && cp -r addon/* /workspace/
python3 calibrate_tau.py --models specxnet mfsanet_A3 mfsanet_A4 mfsanet_v3 --seeds 42 43 44
```
النتائج: results/tau_calibration.csv و results/tau_sensitivity.csv
