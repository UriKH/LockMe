import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from facenet_pytorch import MTCNN
import cv2 as cv
import os
import matplotlib.pyplot as plt

from utils.logger import Logger
from dataset import ModelDataset
from SNN import Net
import config
import model_utils as utils


def train(net, train_loader: DataLoader, valid_loader: DataLoader,
          optimizer: optim, scheduler, criterion, epochs: int, checkpoint_interval: int = 5):
    """
    Train the network
    :param net: the network to train
    :param train_loader: training images loader
    :param valid_loader: validation images loader
    :param optimizer: the chosen optimizer for training
    :param scheduler: learning rate scheduler
    :param criterion: the loss function
    :param epochs: number of epochs to train
    :param checkpoint_interval: number of epochs between checkpoints
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    suffix = 0

    for model in [file for file in os.listdir('checkpoints') if os.path.isfile(os.path.join(os.getcwd(), 'checkpoints', file))]:
        if int(model.split('_')[-1].split('.')[0]) > suffix:
            suffix = int(model.split('_')[-1].split('.')[0])
    print('=== Training started ===')
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        print('---running on train---')
        for i, (img0, img1, labels) in enumerate(train_loader, 0):
            img0, img1, labels = img0.to(device), img1.to(device), labels.to(device)

            optimizer.zero_grad()
            outs = net(img0, img1)
            predicted_labels = (outs >= 0.5).long()

            loss = criterion(outs.squeeze(), labels.view(-1, 1).float().squeeze())
            loss.backward()
            optimizer.step()
            scheduler.step()    # Update the learning rate

            train_loss += loss.item()
            train_correct += (predicted_labels.squeeze() == labels.squeeze()).sum().item()
            total_train += labels.size(0)

            if i % 5 == 0:
                train_accuracy = 100.0 * train_correct / total_train
                print(f'\tcurrent loss: {loss.item()}, current accuracy: {train_accuracy}')

        # calculate training accuracy and training loss
        train_accuracy = 100.0 * train_correct / total_train
        train_loss /= len(train_loader)

        # evaluate the model on the validation set
        net.eval()
        valid_loss = 0.0
        valid_correct = 0
        total_valid = 0
        with torch.no_grad():
            print('---running on validation---')
            for i, (img0, img1, labels) in enumerate(valid_loader):
                img0, img1, labels = img0.to(device), img1.to(device), labels.to(device)

                outs = net(img0, img1)
                predicted_labels = (outs >= 0.5).long()

                loss = criterion(outs.squeeze(), labels.view(-1, 1).float().squeeze())
                valid_loss += loss.item()
                valid_correct += (predicted_labels.squeeze() == labels.squeeze()).sum().item()
                total_valid += labels.size(0)

                if i % 5 == 0:
                    valid_accuracy = 100.0 * valid_correct / total_valid
                    print(f'\tcurrent loss: {loss.item()}, current accuracy: {valid_accuracy}')

        # calculate validation accuracy and training loss
        valid_accuracy = 100.0 * valid_correct / total_valid
        valid_loss /= len(valid_loader)

        print(f'Epoch [{epoch + 1}/{epochs}] - '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
              f'Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%')

        # Append the values for plotting
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(os.getcwd(), 'checkpoints', f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save(net.state_dict(), checkpoint_path)
            print(f'Saved checkpoint at epoch {epoch + suffix + 1}: {checkpoint_path}')

    torch.save(net.state_dict(), config.OUT_PATH)

    # Plotting the results
    plt.figure(figsize=(10, 4))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train')
    plt.plot(range(1, epochs + 1), valid_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train')
    plt.plot(range(1, epochs + 1), valid_accuracies, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def try_it(path=None):
    """
    Utility function for experiencing the network
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    # load the saved parameters
    if path is None:
        state_dict = torch.load(config.MODEL_PATH)
    else:
        state_dict = torch.load(path)

    net = Net().to(device)
    net.load_state_dict(state_dict)  # load the state dictionary into the BCE_and_ContrastiveLoss

    cap = cv.VideoCapture(0)
    frame1 = None

    # capture two images using the keys 'a' and 'b'
    # then calculates the similarity score (if the score is less than 0.5 the images are similar)
    while True:
        ret, frame = cap.read()
        frame = frame[150:350, 150:350]
        frame = cv.resize(frame, (500, 500))
        frame = cv.flip(frame, 1)
        cv.imshow('cam', frame)

        key = cv.waitKey(1)
        if key == ord('a'):
            boxes, conf = mtcnn.detect(frame)
            if boxes is None:
                print('try again')
                continue
            boxes = boxes.astype(int)
            boxes = [box for i, box in enumerate(boxes) if conf[i] >= 0.85]

            frame1 = ModelDataset.create_image(frame, boxes[0])
            frame1 = cv.cvtColor(frame1, cv.COLOR_GRAY2BGR)
        if key == ord('b') and frame1 is not None:
            boxes, conf = mtcnn.detect(frame)
            boxes = boxes.astype(int)
            boxes = [box for i, box in enumerate(boxes) if conf[i] >= 0.85]

            frame2 = ModelDataset.create_image(frame, boxes[0])
            frame2 = cv.cvtColor(frame2, cv.COLOR_GRAY2BGR)

            frame1 = Net.preprocess_image(frame1)
            frame2 = Net.preprocess_image(frame2)
            frame1 = frame1.unsqueeze(0).to(device)
            frame2 = frame2.unsqueeze(0).to(device)

            dist = net(frame1, frame2)
            print(dist)
            frame1 = None
        elif key == ord('q'):
            break


def train_parent():
    """
    Prepare all parameters for training
    """
    # define data transformation
    transformation = transforms.Compose([
        transforms.Resize(config.INPUT_SIZE),
        transforms.ToTensor()
    ])

    # load the dataset for training and validation
    train_ds = ModelDataset(root=config.TRAIN_DATASET_PATH, transform=transformation)
    valid_ds = ModelDataset(root=config.TEST_DATASET_PATH, transform=transformation)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=utils.get_workers())
    valid_loader = DataLoader(valid_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=utils.get_workers())

    Logger('Data loaded', Logger.info).log()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Net().to(device)

    # load the saved parameters
    if config.FINE_TUNE:
        state_dict = torch.load(config.MODEL_PATH)
        net.load_state_dict(state_dict)  # load the state dictionary into the BCE_and_ContrastiveLoss

    criterion = nn.BCEWithLogitsLoss()   # use binary cross entropy as the loss function
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)   # using the Adam optimizer
    scheduler = StepLR(optimizer, step_size=2, gamma=0.99)  # Define the step defined scheduler
    summary(Net(), input_size=[config.INPUT_SHAPE, config.INPUT_SHAPE])

    train(net, train_loader, valid_loader, optimizer, scheduler, criterion, config.EPOCHS, checkpoint_interval=2)


if __name__ == '__main__':
    train_parent()
    try_it()