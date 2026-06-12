"""
Modelli per il task Cats and Dogs: ResNet18 ed EfficientNet-B0.
Progetto 2 - Tecniche di Programmazione Avanzata ed AI
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# --- MODELLO 1: ResNet18 ---
class ResNetCatsDogs(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetCatsDogs, self).__init__()
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            self.resnet = resnet18(weights=weights)
        else:
            self.resnet = resnet18()
            
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.resnet(x)


# --- MODELLO 2: EfficientNet-B0 ---
class EfficientNetCatsDogs(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetCatsDogs, self).__init__()
        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
            self.efficientnet = efficientnet_b0(weights=weights)
        else:
            self.efficientnet = efficientnet_b0()
            
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.efficientnet(x)


if __name__ == "__main__":
    m1 = ResNetCatsDogs(pretrained=True)
    m2 = EfficientNetCatsDogs(pretrained=True)
    print("Entrambi i modelli pre-addestrati (ResNet ed EfficientNet) sono pronti!")