import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        euclidean_dist = F.pairwise_distance(out1, out2, keepdim=True)
        loss = torch.mean((1 - label) * torch.pow(euclidean_dist, 2) +
                          label * torch.pow(torch.clamp(self.margin - euclidean_dist, min=0.0), 2))
        return loss


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Convolutional blocks
        self.cnn = nn.Sequential(
            # nn.Conv2d(1, 64, kernel_size=10),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
            # nn.Dropout(.3),
            #
            # nn.Conv2d(64, 128, kernel_size=7),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
            # nn.Dropout(.3),
            #
            # nn.Conv2d(128, 128, kernel_size=4),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
            # nn.Dropout(.3),

            # Model.conv_block(in_channels=1, out_channels=64, kernel_size=10, dropout=.3),
            # Model.conv_block(in_channels=64, out_channels=128, kernel_size=7, dropout=.3),
            # Model.conv_block(in_channels=128, out_channels=128, kernel_size=4, dropout=.3),
            # Model.conv_block(in_channels=128, out_channels=256, kernel_size=4, max_pool2d=False)

            # nn.Conv2d(128, 256, kernel_size=4),
            # nn.ReLU(inplace=True)

            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            # nn.Dropout(0), # epochs 0 - 70 with dropout 0.1

            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # nn.Dropout(0),# epochs 0 - 70 with dropout 0.1

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        # fully connected part for embedding generation
        self.fc = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            # # nn.ReLU(inplace=True),
            # nn.Linear(256 * 6 * 6, 512),
            # nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size, dropout=0., max_pool2d=True):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
        )
        if max_pool2d:
            block = nn.Sequential(
                block,
                nn.MaxPool2d(2),
                nn.Dropout(dropout)
            )
        else:
            block = nn.Sequential(
                block,
                nn.Dropout(dropout)
            )
        return block

    def forward_once(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

    def forward(self, input1, input2):
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        return out1, out2
