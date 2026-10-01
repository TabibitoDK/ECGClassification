import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, num_classes = 5):
        super(VGG16, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2), 

            nn.Conv2d(64, 128, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(128, 256, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2), 

            nn.Conv2d(256, 512, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2), 

            nn.Conv2d(512, 512, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, num_classes) 
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)