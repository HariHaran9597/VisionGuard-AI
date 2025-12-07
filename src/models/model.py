import torch
import torch.nn as nn

class VisionGuardModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(VisionGuardModel, self).__init__()
        # Load DINOv2 (The "Eye")
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # Classification Head (The "Brain")
        self.head = nn.Sequential(
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits