import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data import DataLoader


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
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            # nn.Dropout(0.01),    # epochs 0 - 70 with dropout 0.1

            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # nn.Dropout(0.01),    # epochs 0 - 70 with dropout 0.1

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        # fully connected part for embedding generation
        self.fc = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.Linear(256, 2)
        )

    def forward_once(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

    def forward(self, input1, input2):
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        return out1, out2


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
        self.max_pool2d = nn.MaxPool2d((2, 2))
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)

        # Define the shared convolutional layers
        self.conv = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.max_pool2d,

            self.dropout1,

            self.conv2,
            self.bn2,
            self.relu,
            self.max_pool2d,

            self.dropout1,

            self.conv3,
            self.bn3,
            self.relu,
            self.max_pool2d,

            self.dropout1,

            self.conv4,
            self.bn4,
            self.relu
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
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


if __name__ == '__main__':
    import shutil
    from facenet_pytorch import MTCNN
    from dataset import ModelDataset
    import cv2 as cv

    def yay(path=r'C:\LockMe_DATA\lfw'):
        new_base = r'C:\LockMe_DATA\LFW_filtered'
        folders = os.listdir(path)
        for folder in folders:
            dir_path = os.path.join(path, folder)
            if len(os.listdir(dir_path)) > 10:
                os.mkdir(os.path.join(new_base, folder))
                for file in os.listdir(dir_path):
                    old_path = os.path.join(dir_path, file)
                    new_path = os.path.join(new_base, folder, file)
                    shutil.copy(old_path, new_path)


    def create_samples_from_folders(parent=r'C:\LockMe_DATA\temp'):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device
        )

        people = os.listdir(parent)
        for i, p in enumerate(people):
            new_dir = os.path.join(parent, f's{i+1}')
            os.makedirs(new_dir)
            for j, file in enumerate(os.listdir(os.path.join(parent, p))):
                image = cv.imread(os.path.join(parent, p, file))
                boxes, conf = mtcnn.detect(image)
                if boxes is None or len(boxes) == 0:
                    print(f'oops, get another image for {os.path.join(parent, p, file)}!')
                    continue
                boxes = boxes.astype(int)
                boxes = [box for i, box in enumerate(boxes)]
                frame = ModelDataset.create_image(image, boxes[0])
                cv.imwrite(os.path.join(new_dir, f'{j+1}.pgm'), frame)

    create_samples_from_folders()

def get_training_accuracy():
    transformation = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])
    ds = ModelDataset(ds_path=r'C:\LockMe_DATA\my_ATNT_DS', transform=transformation)
    test_dataloader = DataLoader(ds, num_workers=2, batch_size=1, shuffle=True)
    dataiter = iter(test_dataloader)
    state_dict = torch.load('ATNT_transformed_SNN_classic_BN_more_data_dropout.pth')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = ClassicModel().to(device)
    net.load_state_dict(state_dict)  # load the state dictionary into the model_torch
    true = 0
    samples = 500
    for i in range(samples):
        x0, x1, label = next(dataiter)
        concatenated = torch.cat((x0, x1), 0)
        out1, out2 = net(x0.to(device), x1.to(device))
        dist = F.pairwise_distance(out1, out2)
        if (dist < 0.9 and label == 0) or (dist > 0.9 and label == 1):
            true += 1.
        # utils.imshow(tv.utils.make_grid(concatenated), f'dissimilarity {dist.item():.2f}')
    print(f'trainset accuracy: {true / samples}')
