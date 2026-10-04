import torch, torch.nn as nn, torch.nn.functional as F

class CCAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.q = nn.Conv2d(c, c//8, 1); self.k = nn.Conv2d(c, c//8, 1); self.v = nn.Conv2d(c, c, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        B,C,H,W = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        q_flat = q.view(B,-1,H*W).permute(0,2,1)
        k_flat = k.view(B,-1,H*W)
        attn = torch.softmax(q_flat @ k_flat, dim=-1)
        v_flat = v.view(B,C,H*W)
        out = (v_flat @ attn.permute(0,2,1)).view(B,C,H,W)
        return x + self.gamma * out

class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc1 = nn.Linear(c, c//r); self.fc2 = nn.Linear(c//r, c)
    def forward(self, x):
        B,C,H,W = x.shape
        z = x.mean(dim=[2,3])
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(z))))
        return x * s.view(B,C,1,1)

class SpatialBranch(nn.Module):
    def __init__(self, input_size=288, out_dim=512):
        super().__init__()
        import timm
        self.backbone = timm.create_model("tf_efficientnetv2_s", pretrained=True,
                                           features_only=True, out_indices=(-1,))
        c = self.backbone.feature_info.channels()[-1]
        self.ccam1, self.ccam2 = CCAM(c), CCAM(c)
        self.se = SEBlock(c, r=16)
        self.mlp = nn.Sequential(nn.Linear(c, out_dim*2), nn.GELU(), nn.Dropout(0.3), nn.Linear(out_dim*2, out_dim))
    def forward(self, x):
        f = self.backbone(x)[-1]
        f = self.ccam2(self.ccam1(f))
        f = self.se(f)
        z = f.mean(dim=[2,3])
        return self.mlp(z)

class DDFCSpectral(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2,64,3,padding=1,stride=2), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64,128,3,padding=1,stride=2), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128,256,3,padding=1,stride=2), nn.BatchNorm2d(256), nn.ReLU(True),
        )
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.mlp = nn.Sequential(nn.Linear(256*4*4, 1024), nn.GELU(), nn.Dropout(0.3), nn.Linear(1024, out_dim))
    def forward(self, x_rgb):
        gray = (0.299*x_rgb[:,0] + 0.587*x_rgb[:,1] + 0.114*x_rgb[:,2]).unsqueeze(1)
        Xf = torch.fft.fft2(gray, norm="ortho")
        mag = torch.log(torch.abs(Xf) + 1e-8)
        phase = torch.angle(Xf)
        mag = (mag - mag.mean(dim=[2,3],keepdim=True)) / (mag.std(dim=[2,3],keepdim=True)+1e-6)
        phase = (phase - phase.mean(dim=[2,3],keepdim=True)) / (phase.std(dim=[2,3],keepdim=True)+1e-6)
        spec = torch.cat([mag, phase], dim=1)
        f = self.conv(spec)
        f = self.pool(f).flatten(1)
        return self.mlp(f)

class CrossAttentionDFA(nn.Module):
    def __init__(self, dim=512, n_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim*2, dim))
        self.ln2 = nn.LayerNorm(dim)
    def forward(self, y_spatial, y_spectral):
        q = y_spatial.unsqueeze(1); kv = y_spectral.unsqueeze(1)
        attn_out, _ = self.attn(q, kv, kv)
        x = self.ln1(q + attn_out)
        x = self.ln2(x + self.ff(x))
        return x.squeeze(1)

class MFSANet(nn.Module):
    def __init__(self, num_classes=2, input_size=288, variant="full"):
        super().__init__()
        self.variant = variant
        if variant in ("spatial_only", "full"):
            self.spatial = SpatialBranch(input_size)
        if variant in ("spectral_only", "full"):
            self.spectral = DDFCSpectral()
        if variant == "full":
            self.fusion = CrossAttentionDFA(dim=512, n_heads=8)
        self.classifier = nn.Sequential(
            nn.Linear(512,256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256,128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        if self.variant == "spatial_only":
            z = self.spatial(x)
        elif self.variant == "spectral_only":
            z = self.spectral(x)
        else:
            z = self.fusion(self.spatial(x), self.spectral(x))
        return self.classifier(z)

class CrossAttentionDFA_Reversed(nn.Module):
    """نفس CrossAttentionDFA، بس معكوسة: الطيفي = Query، المكاني = Key/Value."""
    def __init__(self, dim=512, n_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim*2, dim))
        self.ln2 = nn.LayerNorm(dim)
    def forward(self, y_spatial, y_spectral):
        q = y_spectral.unsqueeze(1); kv = y_spatial.unsqueeze(1)
        attn_out, _ = self.attn(q, kv, kv)
        x = self.ln1(q + attn_out)
        x = self.ln2(x + self.ff(x))
        return x.squeeze(1)


class MFSANetReversed(MFSANet):
    """نفس MFSA-Net بالضبط، فرق واحد: اتجاه الانتباه معكوس (طيفي=Query)."""
    def __init__(self, num_classes=2, input_size=288):
        super().__init__(num_classes=num_classes, input_size=input_size, variant="full")
        self.fusion = CrossAttentionDFA_Reversed(dim=512, n_heads=8)

class MFSANetBranchDrop(MFSANet):
    def __init__(self, num_classes=2, input_size=288, p_drop=0.15):
        super().__init__(num_classes=num_classes, input_size=input_size, variant="full")
        self.p_drop = p_drop

    def forward(self, x):
        ys, yf = self.spatial(x), self.spectral(x)
        if self.training:
            r = torch.rand(ys.size(0), 1, device=ys.device)
            drop_spatial = (r < self.p_drop).float()
            drop_spectral = ((r >= self.p_drop) & (r < 2 * self.p_drop)).float()
            ys = ys * (1 - drop_spatial)
            yf = yf * (1 - drop_spectral)
        return self.classifier(self.fusion(ys, yf))

class MFSANetBranchDrop(MFSANet):
    def __init__(self, num_classes=2, input_size=288, p_drop=0.15):
        super().__init__(num_classes=num_classes, input_size=input_size, variant="full")
        self.p_drop = p_drop

    def forward(self, x):
        ys, yf = self.spatial(x), self.spectral(x)
        if self.training:
            r = torch.rand(ys.size(0), 1, device=ys.device)
            drop_spatial = (r < self.p_drop).float()
            drop_spectral = ((r >= self.p_drop) & (r < 2 * self.p_drop)).float()
            ys = ys * (1 - drop_spatial)
            yf = yf * (1 - drop_spectral)
        return self.classifier(self.fusion(ys, yf))

class MFSANetAuxHeads(MFSANet):
    def __init__(self, num_classes=2, input_size=288, beta=0.3):
        super().__init__(num_classes=num_classes, input_size=input_size, variant="full")
        self.beta = beta
        self.aux_spatial = nn.Linear(512, num_classes)
        self.aux_spectral = nn.Linear(512, num_classes)

    def forward_with_aux(self, x):
        ys, yf = self.spatial(x), self.spectral(x)
        fused_logits = self.classifier(self.fusion(ys, yf))
        return fused_logits, self.aux_spatial(ys), self.aux_spectral(yf)

    def forward(self, x):
        ys, yf = self.spatial(x), self.spectral(x)
        return self.classifier(self.fusion(ys, yf))

class GatedFusion(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, y_spatial, y_spectral):
        g = torch.sigmoid(self.gate(torch.cat([y_spatial, y_spectral], dim=1)))
        g = 0.1 + 0.8 * g
        return g * y_spatial + (1 - g) * y_spectral


class MFSANetGated(MFSANet):
    def __init__(self, num_classes=2, input_size=288):
        super().__init__(num_classes=num_classes, input_size=input_size, variant="full")
        self.fusion = GatedFusion(dim=512)

    def forward(self, x):
        ys, yf = self.spatial(x), self.spectral(x)
        return self.classifier(self.fusion(ys, yf))

class GatedFusion(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, y_spatial, y_spectral):
        g = torch.sigmoid(self.gate(torch.cat([y_spatial, y_spectral], dim=1)))
        g = 0.1 + 0.8 * g
        return g * y_spatial + (1 - g) * y_spectral


class MFSANetGated(MFSANet):
    def __init__(self, num_classes=2, input_size=288):
        super().__init__(num_classes=num_classes, input_size=input_size, variant="full")
        self.fusion = GatedFusion(dim=512)

    def forward(self, x):
        ys, yf = self.spatial(x), self.spectral(x)
        return self.classifier(self.fusion(ys, yf))
