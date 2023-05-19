import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
from torchsummary import summary
import torch.nn.functional as F

from logger import Logger
import model.model_utils as utils
from model.model import ClassicModel
from model.dataset import ModelDataset
import model.config as config


def train(model, loader: DataLoader, optimizer: optim, criterion, epochs=10, ckpt=10):
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
            dist = F.pairwise_distance(out0, out1)
            label = label.view(-1, 1).float()
            loss = criterion(dist.squeeze(), label.squeeze())  # calculate loss

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
    # Save the BCE_and_ContrastiveLoss parameters to a file
    torch.save(model.state_dict(), 'BCE1_C_BN-51_60_lr0003-vec512.pth')
    return train_losses


if __name__ == '__main__':
    transformation = transforms.Compose([transforms.Resize(config.INPUT_SIZE), transforms.ToTensor()])

    ds = ModelDataset(ds_path=config.DATASET_PATH, transform=transformation)
    loader = DataLoader(ds, shuffle=True, num_workers=utils.get_workers(), batch_size=config.BATCH_SIZE)
    Logger('Data loaded', Logger.info).log()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load the saved parameters
    state_dict = torch.load(config.MODEL_PATH)

    net = ClassicModel().to(device)
    net.load_state_dict(state_dict)  # load the state dictionary into the BCE_and_ContrastiveLoss

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)
    summary(ClassicModel(), input_size=[config.INPUT_SHAPE, config.INPUT_SHAPE])

    train(net, loader, optimizer, criterion, epochs=config.EPOCHS)
