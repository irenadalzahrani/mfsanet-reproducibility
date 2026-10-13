# التجارب الإضافية (F3-Net + تجربة موازنة الفئات)
1) مرة وحدة على أي Pod (الملفات على الـVolume المشترك):
   cd /workspace && unzip -o mfsanet_extras_all.zip > /dev/null && cp -r addon/* /workspace/ && chmod +x /workspace/*.sh
2) F3-Net على Pod مستقل:               bash /workspace/s20_f3net.sh
3) موازنة الفئات على Pod مستقل:         bash /workspace/s30_balance.sh nosampler     (أو nofw على Pod آخر)
4) بعد انتهاء الكل:                     bash /workspace/s31_extras_into_finish.sh
