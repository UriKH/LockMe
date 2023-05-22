import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
from torchsummary import summary
import cv2 as cv
import os
import matplotlib.pyplot as plt

from logger import Logger
from dataset import ModelDataset
from model.SNN import Net
import model.config as config


def train(net, train_loader: DataLoader, valid_loader: DataLoader,
          optimizer: optim, criterion, epochs: int, checkpoint_interval: int = 5):
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
            # distances = F.pairwise_distance(embeddings0, embeddings1)
            predicted_labels = (outs >= 0.5).long()

            # loss = criterion(distances.squeeze(), labels.view(-1, 1).float().squeeze())
            loss = criterion(outs.squeeze(), labels.view(-1, 1).float().squeeze())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (predicted_labels.squeeze() == labels.squeeze()).sum().item()
            total_train += labels.size(0)

            if i % checkpoint_interval == 0:
                train_accuracy = 100.0 * train_correct / total_train
                print(f'\tcurrent loss: {loss.item()}, current accuracy: {train_accuracy}')

        train_accuracy = 100.0 * train_correct / total_train
        train_loss /= len(train_loader)

        net.eval()
        valid_loss = 0.0
        valid_correct = 0
        total_valid = 0

        with torch.no_grad():
            print('---running on validation---')
            for i, (img0, img1, labels) in enumerate(valid_loader):
                img0, img1, labels = img0.to(device), img1.to(device), labels.to(device)

                outs = net(img0, img1)
                # embeddings0, embeddings1 = net(img0, img1)
                # distances = F.pairwise_distance(embeddings0, embeddings1)
                predicted_labels = (outs >= 0.5).long()

                loss = criterion(outs.squeeze(), labels.view(-1, 1).float().squeeze())

                valid_loss += loss.item()
                valid_correct += (predicted_labels.squeeze() == labels.squeeze()).sum().item()
                total_valid += labels.size(0)

                if i % checkpoint_interval == 0:
                    valid_accuracy = 100.0 * valid_correct / total_valid
                    print(f'\tcurrent loss: {loss.item()}, current accuracy: {valid_accuracy}')

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
        if (epoch + 1) % 2 == 0:
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
    from facenet_pytorch import MTCNN

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
    while True:
        ret, frame = cap.read()
        frame = frame[150:350, 150:350]
        frame = cv.resize(frame, (500, 500))
        frame = cv.flip(frame, 1)
        cv.imshow('cam', frame)

        key = cv.waitKey(1)
        if key == ord('a'):
            boxes, conf = mtcnn.detect(frame)
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
    transformation = transforms.Compose([transforms.Resize(config.INPUT_SIZE), transforms.ToTensor()])
    ds = ModelDataset(root=config.DATASET_PATH, transform=transformation)
    train_loader, valid_loader = ds.split_dataset()
    Logger('Data loaded', Logger.info).log()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Net().to(device)

    # load the saved parameters
    if config.FINE_TUNE:
        state_dict = torch.load(config.MODEL_PATH)
        net.load_state_dict(state_dict)  # load the state dictionary into the BCE_and_ContrastiveLoss

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)
    summary(Net(), input_size=[config.INPUT_SHAPE, config.INPUT_SHAPE])

    train(net, train_loader, valid_loader, optimizer, criterion, config.EPOCHS)


if __name__ == '__main__':
    train_parent()
    try_it()
