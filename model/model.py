"""
The Siamese Neural Network (SNN) model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from PIL import Image
import cv2 as cv
import numpy as np

from model.dataset import ModelDataset
import model.config as config


class ClassicModel(nn.Module):
    def __init__(self):
        super(ClassicModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=10)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)

        self.max_pool2d = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU(inplace=True)

        # Define the shared convolutional layers
        self.conv = nn.Sequential(
            # Conv block I
            self.conv1, self.bn1, self.relu, self.max_pool2d,
            self.dropout1,

            # Conv block II
            self.conv2, self.bn2, self.relu, self.max_pool2d,
            self.dropout1,

            # Conv block III
            self.conv3, self.bn3, self.relu, self.max_pool2d,
            self.dropout1,

            # Conv block IV
            self.conv4, self.bn4, self.relu
        )

        # Calculate the output size of the conv layers
        self.conv_output_size = self._get_conv_output_size()

        # Define the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.Sigmoid()
        )

        self.fc1 = nn.Linear(self._get_conv_output_size(), 512)

    def _get_conv_output_size(self):
        with torch.no_grad():
            output = self.conv(torch.zeros(1, 1, 100, 100))
        return output.view(-1).size(0)

    def forward_once(self, x):
        """
        :param x: one image sample
        :return: the sample's embedding
        """
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        """
        Automatically called when doing forward propagation
        :param input1: first sample
        :param input2: second sample
        :return: the embeddings of the samples
        """
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

    @staticmethod
    def get_training_accuracy(threshold=0.5):
        """
        Test the model on a portion of the training set to get useful metrics
        :param threshold: The positive/ negative distance threshold
        """
        transformation = transforms.Compose([
            transforms.Resize(config.INPUT_SIZE),
            transforms.ToTensor()
        ])

        ds = ModelDataset(ds_path=config.DATASET_PATH, transform=transformation)
        test_dataloader = DataLoader(ds, num_workers=2, batch_size=1, shuffle=True)
        dataiter = iter(test_dataloader)
        state_dict = torch.load(config.MODEL_PATH)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net = ClassicModel().to(device)
        net.load_state_dict(state_dict)  # load the state dictionary into the BCE_and_ContrastiveLoss

        true = 0
        samples = 500
        positives, negatives = [], []
        for i in range(samples):
            x0, x1, label = next(dataiter)
            out1, out2 = net(x0.to(device), x1.to(device))
            dist = F.pairwise_distance(out1, out2)
            if (dist < threshold and label == 0) or (dist > threshold and label == 1):
                true += 1.
                if label == 0:
                    positives.append(dist)
                else:
                    negatives.append(dist)
        above_mean = np.sum(np.mean(torch.cat(positives).detach().numpy()) < torch.cat(positives).detach().numpy())
        bellow_mean = np.sum(np.mean(torch.cat(positives).detach().numpy()) > torch.cat(positives).detach().numpy())
        print(f'train set accuracy: {true / samples}')
        print(f'positives:')
        print(f'mean: {np.mean(torch.cat(positives).detach().numpy()):.3f}\t'
              f'above: {above_mean:.3f}, bellow: {bellow_mean:.3f}\n'
              f'median: {np.median(torch.cat(positives).detach().numpy()):.3f}\t'
              f'minimum {np.min(torch.cat(positives).detach().numpy()):.3f}\t'
              f'maximum: {np.max(torch.cat(positives).detach().numpy()):.3f}\n')
        above_mean = np.sum(np.mean(torch.cat(negatives).detach().numpy()) < torch.cat(negatives).detach().numpy())
        bellow_mean = np.sum(np.mean(torch.cat(negatives).detach().numpy()) > torch.cat(negatives).detach().numpy())
        print(f'negatives:')
        print(f'mean: {np.mean(torch.cat(negatives).detach().numpy()):.3f}\t'
              f'above: {above_mean:.3f}, bellow: {bellow_mean:.3f}\n'
              f'median: {np.median(torch.cat(negatives).detach().numpy()):.3f}\t'
              f'minimum {np.min(torch.cat(negatives).detach().numpy()):.3f}\t'
              f'maximum: {np.max(torch.cat(negatives).detach().numpy()):.3f}')

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
