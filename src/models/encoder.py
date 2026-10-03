import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class EncoderCNN(nn.Module):
    def __init__(self, fine_tune=False):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Drop the final avgpool + fc (classification head) — we want the
        # spatial feature map, not a 1000-class prediction.
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.fine_tune(fine_tune)

    def forward(self, images):
        features = self.resnet(images)                      # [B, 2048, 7, 7]
        features = features.permute(0, 2, 3, 1)              # [B, 7, 7, 2048]
        features = features.view(features.size(0), -1, features.size(-1))  # [B, 49, 2048]
        return features

    def fine_tune(self, fine_tune=False):
        """Frozen by default (Phase 4 baseline). Set True only for a later
        Phase 6 experiment comparing frozen vs. fine-tuned encoders."""
        for p in self.resnet.parameters():
            p.requires_grad = False
        if fine_tune:
            for child in list(self.resnet.children())[7:]:   # unfreeze last block only
                for p in child.parameters():
                    p.requires_grad = True