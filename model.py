import torch.nn as nn
from torchvision import models

def get_ENB0relu():
    model_ft = models.efficientnet_b0(weights=None)
    model_ft = model_ft.children()
    model_ft = nn.Sequential(*model_ft)
    model_ft[-1][-1] = nn.Conv2d(model_ft[-1][-1].in_features, 1, kernel_size=1, stride=1)
    model_ft = nn.Sequential(model_ft, nn.ReLU())
    return model_ft
