import torch.nn as nn
import torchvision.models as models

class ResNet9(nn.Module):
    def __init__(self, num_classes):
        super(ResNet9, self).__init__()
        self.model = models.resnet18(pretrained=True)
        # Gantilah 'fc' layer agar sesuai dengan jumlah kelas yang diinginkan
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
