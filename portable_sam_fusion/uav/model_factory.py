import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.output_shapes = [
            (1, 256, 128, 128),
            (1, 512, 64, 64),
            (1, 1024, 32, 32),
            (1, 2048, 16, 16),
        ]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return features


def create_resnet_backbone(pretrained: bool = True) -> ResNetBackbone:
    if pretrained:
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        resnet = models.resnet50(weights=None)
    for m in resnet.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    return ResNetBackbone(resnet)

