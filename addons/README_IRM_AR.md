# تجربة IRM (على Pod فاضي)
cd /workspace && unzip -o mfsanet_irm_addon.zip > /dev/null && cp -r addon/* /workspace/ && chmod +x /workspace/*.sh
bash /workspace/s33_irm.sh          # اختبار سريع -> تدريب 3 بذور -> تقييم -> فحص التصفير (آخره IRM_DONE)
bash /workspace/s34_irm_compare.sh  # المقارنة مع A3 (بعد s14)
