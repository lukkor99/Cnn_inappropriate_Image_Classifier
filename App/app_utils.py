import torch
import torchvision
import torch.nn as nn
from PIL import Image
import io

num_classes = 3


# model class
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 7, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 768),
            nn.ReLU(True),
            nn.Linear(768, 128),
            nn.ReLU(True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# model init
model = Network()
model.load_state_dict(torch.load('model.pth',map_location=torch.device('cpu')))
model.eval()


# image transforms
def transform(image_bytes):
    my_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(512),
        torchvision.transforms.CenterCrop(512),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


# predicting function
def get_predict(image):
    images = image.reshape(-1, 512 * 512)
    output = model.forward(image)

    _, predicted = torch.max(output.data, 1)
    return predicted
