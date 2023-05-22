"""
The Siamese Neural Network (SNN) model
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2 as cv
import model.config as config


class Net(nn.Module):
    embedding_size = 4096

    def __init__(self):
        super(Net, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 64, 10),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.0)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 7),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.0)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 128, 4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.0)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, 4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.Sigmoid()
        )

        self.fcOut = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()

    def forward_once(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)

        x = torch.abs(x1 - x2)
        x = self.fcOut(x)
        return x

    def forward_embeddings(self, x1, x2):
        x = torch.abs(x1 - x2)
        x = self.fcOut(x)
        return x

    @staticmethod
    def preprocess_image(image):
        """
        Preprocess the image before inserting it to the network
        :param image: the sample
        :return: the new image in the model's format
        """
        image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2GRAY)).convert('L')

        transform = transforms.Compose([
            transforms.Resize(config.INPUT_SIZE),
            transforms.ToTensor()
        ])

        image = transform(image)
        return image
