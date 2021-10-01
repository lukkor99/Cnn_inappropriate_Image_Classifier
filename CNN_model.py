import torch
import torchvision
import torch.nn as nn
import numpy as np
from torchvision import datasets
import os

# gpu support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# batch size and folder structure
struct = ['train', 'val']
batch = 30

# data augmentation
transforms = {'train': torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(512),
    torchvision.transforms.RandomAffine(15 ,(0.30, 0.30)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]),
    'val': torchvision.transforms.Compose([
        torchvision.transforms.Resize(512),
        torchvision.transforms.CenterCrop(512),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
}

# paths
directory = 'data'
PATH = '../model.pth'

image_set = {x: datasets.ImageFolder(os.path.join(directory, x),
                                     transforms[x])
             for x in struct
             }

image_loader = {x: torch.utils.data.DataLoader(image_set[x], batch_size=batch,
                                               num_workers=0)
                for x in struct
                }

# datasets
train_set = image_set['train']
test_set = image_set['val']

# loaders
train_loader = {x: torch.utils.data.DataLoader(image_set[x], batch_size=batch,
                                               num_workers=0, shuffle=True)
                for x in struct}
train_loader = train_loader['train']
test_loader = image_loader['val']

# training parameters
sizes = {len(train_set)}
classes = train_set.classes
num_classes = len(classes)

inputs, names = next(iter(train_loader))


# model build
class CnnModel(nn.Module):

    def __init__(self):
        super(CnnModel, self).__init__()
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


model = CnnModel().to(device)

# hyper parameters
epochs = 70
learning_rate = 0.0003
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.5)


steps = len(train_loader)
loss_array = []
epoch_loss = []

# training and validating loop
for epoch in range(epochs):
    model.train()
    total_train = 0
    correct_train = 0
    running_loss = 0
    # training
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        loss_array.append(loss.item())
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.nelement()
        correct_train += predicted.eq(labels.data).sum().item()
        train_accuracy = 100 * correct_train / total_train

    epoch_loss.append(np.mean(loss_array))

    model.eval()
    # validation
    with torch.no_grad():
        eval_loss = []
        e_loss = 0
        correct = 0
        samples = 0
        class_correct = [0 for i in range(num_classes)]
        samples_correct = [0 for i in range(num_classes)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            e_loss = criterion(outputs, labels)

            eval_loss.append(e_loss.item())
            samples += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(batch):
                label = labels[i]
                pred = predicted[i]

                if label == pred:
                    class_correct[label] += 1
                samples_correct[label] += 1

        acc = 100.0 * correct / samples
        # model performance
        print(f'epoch {epoch + 1}/{epochs} train accuracy = {train_accuracy:.2f}% loss = {epoch_loss[epoch]:.3f}'
              f' eval accuracy = {acc:.2f}% loss = {np.mean(eval_loss):.3f}')

    loss_array.clear()
    scheduler.step()
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'model{epoch}.pth')

torch.save(model.state_dict(), PATH)

# output in file : epochs.txt.txt
