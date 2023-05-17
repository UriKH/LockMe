import torch
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
from torchsummary import summary
import torch.nn.functional as F

import multiprocessing
from tqdm import tqdm
import cv2 as cv
from PIL import Image


from dataset import ModelDataset
from logger import Logger
import utils
from model import Model, ContrastiveLoss, ClassicModel


def get_workers():
    num_workers = multiprocessing.cpu_count()
    if num_workers > 1:
        num_workers -= 1
    return num_workers


def preprocess_image(image, input_size):
    image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2GRAY)).convert('L')

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor()
    ])

    image = transform(image)
    return image


def train(model, loader: DataLoader, optimizer: optim, criterion: ContrastiveLoss, epochs=10, ckpt=10):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cntr = []
    loss_hist = []
    iteration = 0

    train_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        print("Starting epoch " + str(epoch + 1))
        for i, (img0, img1, label) in enumerate(loader, 0):
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            out0, out1 = model(img0, img1)  # forward-prop
            loss = criterion(out0, out1, label)  # calculate loss

            optimizer.zero_grad()   # zero the gradients
            loss.backward()     # calculate back-prop
            optimizer.step()    # optimize
            running_loss += loss.item()

            if i % ckpt == 0:
                Logger(f'\nEpoch {epoch}:\n\tloss: {loss.item()}', Logger.info).log()
                iteration += ckpt
                cntr.append(iteration)
                loss_hist.append(loss.item())
        avg_train_loss = running_loss / len(loader)
        train_losses.append(avg_train_loss)
        val_running_loss = 0.0
    utils.show_plot(cntr, loss_hist)

    print("Finished Training")
    # Save the model_torch parameters to a file
    torch.save(model.state_dict(), 'ATNT_transformed_SNN_classic_BN_more_data_dropout3.pth')
    return train_losses


def main():
    ds_path = r'C:\LockMe_DATA\temp'
    transformation = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])

    ds = ModelDataset(ds_path=ds_path, transform=transformation)
    loader = DataLoader(ds, shuffle=True, num_workers=get_workers(), batch_size=128)
    Logger('Data loaded', Logger.info).log()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load the saved parameters
    state_dict = torch.load('ATNT_transformed_SNN_classic_BN_more_data_dropout2.pth')

    net = ClassicModel().to(device)
    # net = Model().to(device)
    net.load_state_dict(state_dict)     # load the state dictionary into the model_torch

    loss = ContrastiveLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.00005)  # 0 - 70 (0.0005) # 70 - 100 (0.001) # 100 - 120 (0.0005)
    summary(ClassicModel(), input_size=[(1, 100, 100), (1, 100, 100)])

    train(net, loader, optimizer, loss, epochs=20)

    test_dataloader = DataLoader(ds, num_workers=2, batch_size=1, shuffle=True)
    dataiter = iter(test_dataloader)
    true = 0
    samples = 500
    for i in range(samples):
        x0, x1, label = next(dataiter)
        concatenated = torch.cat((x0, x1), 0)
        out1, out2 = net(x0.to(device), x1.to(device))
        dist = F.pairwise_distance(out1, out2)
        if (dist < 0.7 and label == 0) or (dist > 0.7 and label == 1):
            true += 1.
        # utils.imshow(tv.utils.make_grid(concatenated), f'dissimilarity {dist.item():.2f}')
    print(f'trainset accuracy: {true/samples}')


def try_it():
    from facenet_pytorch import MTCNN

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    # load the saved parameters
    state_dict = torch.load('ATNT_transformed_SNN_classic_BN_more_data_dropout2.pth')

    net = ClassicModel().to(device)
    net.load_state_dict(state_dict)  # load the state dictionary into the model_torch

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
            boxes = [box for i, box in enumerate(boxes) if conf[i] >= 0.95]

            frame1 = ModelDataset.create_image(frame, boxes[0])
            frame1 = cv.cvtColor(frame1, cv.COLOR_GRAY2BGR)
        if key == ord('b') and frame1 is not None:
            boxes, conf = mtcnn.detect(frame)
            boxes = boxes.astype(int)
            boxes = [box for i, box in enumerate(boxes) if conf[i] >= 0.95]

            frame2 = ModelDataset.create_image(frame, boxes[0])
            frame2 = cv.cvtColor(frame2, cv.COLOR_GRAY2BGR)

            frame1 = preprocess_image(frame1, (100, 100))
            frame2 = preprocess_image(frame2, (100, 100))
            frame1 = frame1.unsqueeze(0).to(device)
            frame2 = frame2.unsqueeze(0).to(device)

            out1, out2 = net(frame1, frame2)
            dist = F.pairwise_distance(out1, out2).item()
            frame1 = None
            print(f'distance: {dist:.3f}')
        elif key == ord('q'):
            break


if __name__ == '__main__':
    # ds_path = r'C:\LockMe_DATA\temp'
    # transformation = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])
    #
    # ds = ModelDataset(ds_path=ds_path, transform=transformation)
    # ds.transfer_dataset(discover=False)
    main()

    # try_it()
