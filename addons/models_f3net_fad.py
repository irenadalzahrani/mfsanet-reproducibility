"""F3-Net baseline = the DeepfakeBench implementation of F3-Net (FAD branch + Xception), loaded AT RUN TIME from a clone of
https://github.com/SCLBD/DeepfakeBench (training/detectors/f3net_detector.py). The FAD code is NOT copied or modified here (that repository is CC BY-NC 4.0):
this file only extracts the classes Filter / FAD_Head and the helpers DCT_mat / generate_filter / norm_sigma from the cloned file and puts them in front of an
ImageNet-pretrained Xception whose first convolution takes the 12 FAD channels. Same head as the Xception baseline, same training protocol as every other model.
Note: the DeepfakeBench file states that it uses ONLY the FAD branch (no LFS stream, no MixBlock) of the original two-stream F3-Net.
    python3 models_f3net_fad.py --selftest"""
import ast, os, sys
import numpy as np
import torch, torch.nn as nn
import timm

DB_FILE = os.environ.get("DEEPFAKEBENCH_F3NET", "/workspace/DeepfakeBench/training/detectors/f3net_detector.py")
WANT = ("Filter", "FAD_Head", "DCT_mat", "generate_filter", "norm_sigma")


def load_fad_head(path=DB_FILE, extra_ns=None):
    tree = ast.parse(open(path, encoding="utf-8").read())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in WANT]
    found = {n.name for n in nodes}
    assert found == set(WANT), f"could not find {sorted(set(WANT) - found)} in {path}"
    ns = {"np": np, "torch": torch, "nn": nn, "F": torch.nn.functional} if extra_ns is None else dict(extra_ns)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)
    return ns["FAD_Head"]


class F3NetFAD(nn.Module):
    def __init__(self, num_classes=2, input_size=288):
        super().__init__()
        self.fad = load_fad_head()(input_size)                                              # DeepfakeBench's FAD_Head: 4 learnable bands x RGB = 12 channels
        self.backbone = timm.create_model("xception", pretrained=True, num_classes=0, in_chans=12)   # first conv adapted from the ImageNet weights
        self.classifier = nn.Sequential(nn.Linear(self.backbone.num_features, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.backbone(self.fad(x.float())))


def selftest(size=288):
    torch.manual_seed(0)
    x = torch.randn(2, 3, size, size)
    fad = load_fad_head()(size); y = fad(x); print("FAD_Head output", tuple(y.shape)); assert y.shape == (2, 12, size, size) and torch.isfinite(y).all()
    m = F3NetFAD(input_size=size); print(f"F3NetFAD parameters: {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M")
    m.train(); z = m(x); print("logits", tuple(z.shape)); assert z.shape == (2, 2)
    z.sum().backward(); print("backbone conv1 weight shape", tuple(m.backbone.conv1.weight.shape), "(expected (32, 12, 3, 3))")
    assert tuple(m.backbone.conv1.weight.shape) == (32, 12, 3, 3)
    print("SELFTEST OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest()
