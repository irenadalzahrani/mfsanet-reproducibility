# A2 (الطيفي فقط) باختيار النقطة حسب AUC
cd /workspace && unzip -o mfsanet_a2auc_addon.zip > /dev/null && cp -r addon/* /workspace/ && chmod +x /workspace/*.sh
bash /workspace/s32_a2_auc.sh
