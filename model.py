import torch.nn as nn
from torchvision import models

'''
def get_model():
    model_ft = models.regnet_y_8gf(pretrained=True)
    # The following is required to allow for single channel input instead of RGB expected input
    model_ft = list(model_ft.children())
    w = model_ft[0][0].weight
    model_ft[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model_ft[0][0].weight = nn.Parameter(torch.mean(w, dim=1, keepdim=True))
    model_ft = nn.Sequential(*model_ft)
    # The following is to change the output neurons to 52, the number of VF points to estimate
    model_ft[-1] = nn.Conv2d(model_ft[-1].in_features, 52, kernel_size=1, stride=1)
    return model_ft
'''

def get_RN50():
    model_ft = models.resnet50(weights=None)
    model_ft = list(model_ft.children())
    model_ft = nn.Sequential(*model_ft)
    # The following is to change the output neurons to a given number, the number of output neurons
    model_ft[-1] = nn.Conv2d(model_ft[-1].in_features, 1, kernel_size=1, stride=1)
    return model_ft

def get_RN50relu():
    model_ft = models.resnet50(weights=None)
    model_ft = list(model_ft.children())
    model_ft = nn.Sequential(*model_ft)
    # The following is to change the output neurons to a given number, the number of output neurons
    model_ft[-1] = nn.Conv2d(model_ft[-1].in_features, 1, kernel_size=1, stride=1)
    model_ft = nn.Sequential(model_ft, nn.ReLU())
    return model_ft

def get_SWv2srelu():
    model_ft = models.swin_v2_s(weights=None)
    model_ft = list(model_ft.children())
    model_ft = nn.Sequential(*model_ft)
    # The following is to change the output neurons to a given number, the number of output neurons
    model_ft[-1] = nn.Linear(model_ft[-1].in_features, 1)
    model_ft = nn.Sequential(model_ft, nn.ReLU())
    return model_ft

def get_ENB0():
    model_ft = models.efficientnet_b0(weights=None)
    model_ft = model_ft.children()
    model_ft = nn.Sequential(*model_ft)
    model_ft[-1][-1] = nn.Conv2d(model_ft[-1][-1].in_features, 1, kernel_size=1, stride=1)
    return model_ft

def get_ENB0relu():
    model_ft = models.efficientnet_b0(weights=None)
    model_ft = model_ft.children()
    model_ft = nn.Sequential(*model_ft)
    model_ft[-1][-1] = nn.Conv2d(model_ft[-1][-1].in_features, 1, kernel_size=1, stride=1)
    model_ft = nn.Sequential(model_ft, nn.ReLU())
    return model_ft

def get_ENB4relu():
    model_ft = models.efficientnet_b4(weights=None)
    model_ft = model_ft.children()
    model_ft = nn.Sequential(*model_ft)
    model_ft[-1][-1] = nn.Conv2d(model_ft[-1][-1].in_features, 1, kernel_size=1, stride=1)
    model_ft = nn.Sequential(model_ft, nn.ReLU())
    return model_ft

def get_ENv2srelu():
    model_ft = models.efficientnet_v2_s(weights=None)
    model_ft = model_ft.children()
    model_ft = nn.Sequential(*model_ft)
    model_ft[-1][-1] = nn.Conv2d(model_ft[-1][-1].in_features, 1, kernel_size=1, stride=1)
    model_ft = nn.Sequential(model_ft, nn.ReLU())
    return model_ft

def get_SWv2sconvrelu():
    model_ft = models.swin_v2_s(weights=None)
    model_ft = list(model_ft.children())
    model_ft = nn.Sequential(*model_ft)
    model_ft[4] = nn.Conv2d(model_ft[5].in_features, 1, kernel_size=1, stride=1)
    model_ft[5] = nn.ReLU()
    return model_ft
