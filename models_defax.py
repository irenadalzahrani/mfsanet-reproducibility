import torch, torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, cs, ce, d=256, n_heads=8):
        super().__init__()
        self.n_heads, self.d, self.hd = n_heads, d, d//n_heads
        self.wq = nn.Linear(cs, d); self.wk = nn.Linear(ce, d)
        self.wv = nn.Linear(ce, d); self.wo = nn.Linear(d, cs)
    def forward(self, swin_tok, eff_tok):
        B,N,_ = swin_tok.shape; _,M,_ = eff_tok.shape
        q = self.wq(swin_tok).view(B,N,self.n_heads,self.hd).transpose(1,2)
        k = self.wk(eff_tok).view(B,M,self.n_heads,self.hd).transpose(1,2)
        v = self.wv(eff_tok).view(B,M,self.n_heads,self.hd).transpose(1,2)
        attn = torch.softmax((q @ k.transpose(-2,-1)) / (self.hd**0.5), dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B,N,self.d)
        return self.wo(out)

class DeFaX(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, input_size=288):
        super().__init__()
        import timm
        self.swin = timm.create_model("swin_base_patch4_window12_384", pretrained=pretrained,
                                       num_classes=0, global_pool="", img_size=input_size)
        self.cs = self.swin.num_features
        self.effnet = timm.create_model("efficientnet_b3", pretrained=pretrained,
                                         features_only=True, out_indices=(4,))
        self.ce = self.effnet.feature_info.channels()[-1]
        self.cross_attn = CrossAttentionFusion(self.cs, self.ce, d=256, n_heads=8)
        self.classifier = nn.Sequential(nn.Linear(self.cs,256), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(256,num_classes))

    def _swin_tokens(self, x):
        f = self.swin.forward_features(x)
        if f.dim()==4:
            B,H,W,C = f.shape; f = f.reshape(B,H*W,C)
        return f
    def _eff_tokens(self, x):
        fmap = self.effnet(x)[-1]; B,C,H,W = fmap.shape
        return fmap.flatten(2).transpose(1,2)

    def forward(self, x):
        swin_tok = self._swin_tokens(x)
        eff_tok = self._eff_tokens(x)
        fused = self.cross_attn(swin_tok, eff_tok)
        z = fused.mean(dim=1)
        return self.classifier(z)
