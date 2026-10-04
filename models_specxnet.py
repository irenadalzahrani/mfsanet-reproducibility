import torch, torch.nn as nn, torch.nn.functional as F

class ChannelSplit(nn.Module):
    def __init__(self, channels, alpha=0.5):
        super().__init__()
        self.c_spec = max(1, int(round(channels*alpha)))
        self.c_spat = channels - self.c_spec
    def forward(self, x): return torch.split(x, [self.c_spat, self.c_spec], dim=1)

class LocalSpatialBranch(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.bn = nn.BatchNorm2d(c)
    def forward(self, x): return F.relu(self.bn(self.conv(x)), inplace=True)

class GlobalSpectralBranch(nn.Module):
    def __init__(self, c, h, w):
        super().__init__()
        fw = w//2 + 1
        self.wr = nn.Parameter(torch.ones(1,c,h,fw))
        self.wi = nn.Parameter(torch.zeros(1,c,h,fw))
        self.bn = nn.BatchNorm2d(c)
    def forward(self, x):
        Xf = torch.fft.rfft2(x, norm="ortho")
        Xf = Xf * torch.complex(self.wr, self.wi)
        y = torch.fft.irfft2(Xf, s=x.shape[-2:], norm="ortho")
        return F.relu(self.bn(y), inplace=True)

class DDFC(nn.Module):
    def __init__(self, c, h, w, alpha=0.5):
        super().__init__()
        self.split = ChannelSplit(c, alpha)
        self.local = LocalSpatialBranch(self.split.c_spat)
        self.glob = GlobalSpectralBranch(self.split.c_spec, h, w)
        self.proj_l = nn.Conv2d(self.split.c_spat, c, 1)
        self.proj_g = nn.Conv2d(self.split.c_spec, c, 1)
    def forward(self, x):
        xl, xg = self.split(x)
        return self.proj_l(self.local(xl)), self.proj_g(self.glob(xg))

class DFA(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.att_g = nn.Linear(c, c)
        self.att_l = nn.Linear(c, c)
        self.fuse = nn.Linear(2*c, 2)
    def forward(self, yl, yg):
        zl, zg = self.gap(yl).flatten(1), self.gap(yg).flatten(1)
        ag = torch.sigmoid(self.att_g(zg)).unsqueeze(-1).unsqueeze(-1)
        al = torch.sigmoid(self.att_l(zl)).unsqueeze(-1).unsqueeze(-1)
        yl_t, yg_t = yl + yl*ag, yg + yg*al
        g = torch.softmax(self.fuse(torch.cat([zl,zg],1)), dim=1)
        gl, gg = g[:,0].view(-1,1,1,1), g[:,1].view(-1,1,1,1)
        return gl*yl_t + gg*yg_t

class SFBlock(nn.Module):
    def __init__(self, c, h, w, alpha=0.5):
        super().__init__()
        self.ddfc = DDFC(c, h, w, alpha)
        self.dfa = DFA(c)
    def forward(self, x):
        yl, yg = self.ddfc(x)
        return x + self.dfa(yl, yg)

class SpecXNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, alpha=0.5, input_size=288):
        super().__init__()
        import timm
        self.backbone = timm.create_model("legacy_xception", pretrained=pretrained,
                                           features_only=True, out_indices=(3,4))
        c3, c4 = self.backbone.feature_info.channels()
        with torch.no_grad():
            f3, f4 = self.backbone(torch.zeros(1,3,input_size,input_size))
            h3,w3 = f3.shape[-2:]; h4,w4 = f4.shape[-2:]
        self.sf3 = SFBlock(c3, h3, w3, alpha)
        self.sf4 = SFBlock(c4, h4, w4, alpha)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(c3+c4,256), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(256,num_classes))
    def forward(self, x):
        f3, f4 = self.backbone(x)
        f3, f4 = self.sf3(f3), self.sf4(f4)
        p3, p4 = self.pool(f3).flatten(1), self.pool(f4).flatten(1)
        return self.classifier(torch.cat([p3,p4],1))
