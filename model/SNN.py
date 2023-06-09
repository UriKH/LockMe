import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2 as cv
import config


class Net(nn.Module):
    """
    The Siamese Neural Network (SNN) model
    """
    embedding_size = 4096

    def __init__(self):
        super(Net, self).__init__()
        # conv block structure:
        # input: 105 x 105 (1 depth)
        # 2D convolution -> batch normalization -> ReLU activation -> max-pooling (2D) -> drop-out

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 64, 10),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.6) # was 0.2
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 7),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.6)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 128, 4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.6)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, 4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Calculate the output size of the conv layers
        self.conv_output_size = self._get_conv_output_size()

        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_size, 4096),
            nn.Sigmoid()
        )

        self.fcOut = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()

    def _get_conv_output_size(self):
        x = self.conv_block1(torch.zeros(1, 1, config.INPUT_SIZE[0], config.INPUT_SIZE[1]))
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        return x.view(x.size(0), -1).size(1)

    def forward_once(self, x):
        """
        Forward propagation through the convolutional part of the SNN
        :param x: the transformed image
        :return: embedding of the transformed image
        """
        # INPUT: 1 ,  105 x 105
        x = self.conv_block1(x)     # 64,  48  x 48
        x = self.conv_block2(x)     # 128, 21  x 21
        x = self.conv_block3(x)     # 128, 9   x 9
        x = self.conv_block4(x)     # 256, 6   x 6
        x = x.view(-1, self.conv_output_size)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        """
        Forward propagation through the SNN
        This function is automatically called
        :param x1: first transformed image
        :param x2: second transformed image
        :return: the prediction of the network
        """
        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)

        x = torch.abs(x1 - x2)
        x = self.fcOut(x)
        return x

    def forward_embeddings(self, x1, x2):
        """
        Run the SNN model on two image embeddings
        :param x1: first face embedding
        :param x2: second face embedding
        :return: the prediction of the network
        """
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
