from torchvision.models import resnet18, resnet50
import torch.nn as nn
import torch

class Std(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18= resnet18(num_classes = 100)            
        self.fc = nn.Linear(512, 2048)
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        x1 = self.fc(x)
        x2 = self.resnet18.fc(x)
        return x1, x2
class Teacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet50 = resnet50(num_classes = 100)
        self.resnet50.load_state_dict(torch.load("teacher_best.pth"))
    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)
        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)
        return x