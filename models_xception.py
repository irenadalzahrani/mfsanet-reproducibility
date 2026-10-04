import torch
import torch.nn as nn
import timm

class XceptionBaseline(nn.Module):
    def __init__(self, num_classes=2, input_size=288):
        super().__init__()
        self.backbone = timm.create_model("xception", pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)
