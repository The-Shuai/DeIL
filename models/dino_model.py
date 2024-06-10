
import torch
import torchvision.models as models

class DinoModel(models.ResNet):
    def __init__(self):
        super(DinoModel, self).__init__(block=models.resnet.Bottleneck, layers=[3, 4, 6, 3], num_classes=1000)
        
        # Remove layer2, layer3, and layer4
        delattr(self, 'layer2')
        delattr(self, 'layer3')
        delattr(self, 'layer4')
        # delattr(self, 'layer5')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class DinoModel_(torch.nn.Module):
    def __init__(self):
        super(DinoModel_, self).__init__()
        
        original_model = models.resnet50(pretrained=True)
        self.layer1 = torch.nn.Sequential(*list(original_model.children())[:6])

    def forward(self, x):
        x = self.layer1(x)
        return x

# Create an instance of the modified DinoModel
dino_model = DinoModel()

# Print the modified model architecture
print(dino_model)