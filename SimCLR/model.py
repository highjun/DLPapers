import torch.nn as nn 
import torch
from torchvision.models import resnet18

def pretrainedModel():
    model = resnet18()
    model.fc = nn.Sequential(
        nn.Linear(512,256),
        nn.ReLU(),
        nn.Linear(256, 128)
    )
    return model

def downstreamModel(n_class, freeze, pretrain = None):
    model = None
    if pretrain is not None:
        model = pretrainedModel()
        model.load_state_dict(pretrain)
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
    else:
        model = resnet18()
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, n_class)
    )
    return model