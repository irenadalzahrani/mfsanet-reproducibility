# إضافة VREx + فحص تصفير الفروع

ارفعي `mfsanet_vrex_addon.zip` إلى /workspace ثم:
```bash
cd /workspace && unzip -o mfsanet_vrex_addon.zip > /dev/null && cp -r addon/* /workspace/ && python3 patch_eval_registry.py
```
1) فحص التصفير (قراءة فقط، بعد تدريب A3 وA4): `python3 zero_branch_check.py --models mfsanet_A3 mfsanet_A4 --datasets ffpp celebdf`
2) اختبار VREx السريع: `python3 train_vrex.py --seeds 42 --quick-debug`
3) التدريب (3 بذور): `nohup python3 train_vrex.py --seeds 42 43 44 > logs/vrex.log 2>&1 &`
4) التقييم: `python3 eval_predictions.py --dataset ffpp --models mfsanet_A3 mfsanet_vrex --seeds 42 43 44` (وكذلك celebdf وdfdc)
5) فحص التصفير على VREx: `python3 zero_branch_check.py --models mfsanet_vrex --datasets ffpp celebdf`
