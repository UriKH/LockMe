import torch
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
from torchsummary import summary
import torch.nn.functional as F

import multiprocessing
from tqdm import tqdm

from dataset import ModelDataset
from logger import Logger
import utils
from model import Model, ContrastiveLoss


def get_workers():
    num_workers = multiprocessing.cpu_count()
    if num_workers > 1:
        num_workers -= 1
    return num_workers


def training(model, loader: DataLoader, optimizer: optim, loss_obj: ContrastiveLoss, epochs=10, ckpt=10):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cntr = []
    loss_hist = []
    iteration = 0

    for epoch in range(epochs):
        for i, (img0, img1, label) in enumerate(loader, 0):
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)

            optimizer.zero_grad()   # zero the gradients
            out0, out1 = model(img0, img1)    # forward-prop
            loss = loss_obj(out0, out1, label)  # calculate loss
            loss.backward()     # calculate back-prop
            optimizer.step()    # optimize

            if i % ckpt == 0:
                Logger(f'\nEpoch {epoch}:\n\tloss: {loss.item()}', Logger.info).log()
                iteration += ckpt
                cntr.append(iteration)
                loss_hist.append(loss.item())
    utils.show_plot(cntr, loss_hist)

    # Save the model_torch parameters to a file
    torch.save(model.state_dict(), 'model_params_3.pth')


def main():
    ds_path = r'C:\LockMe_DATA\ATNT'
    transformation = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])

    ds = ModelDataset(ds_path=ds_path, transform=transformation)
    loader = DataLoader(ds, shuffle=True, num_workers=get_workers(), batch_size=64)
    Logger('Data loaded', Logger.info).log()

    # --- example data batch ---
    # example_batch = next(iter(loader))
    # concat = torch.cat((example_batch[0], example_batch[1]), 0)
    # utils.imshow(tv.utils.make_grid(concat))
    # print(example_batch[2].numpy().reshape(-1))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load the saved parameters
    state_dict = torch.load('model_params_3.pth')

    net = Model().to(device)
    net.load_state_dict(state_dict)     # load the state dictionary into the model_torch

    # loss = ContrastiveLoss().to(device)
    # optimizer = optim.Adam(net.parameters(), lr=0.00025)  # 0 - 70 (0.0005) # 70 - 100 (0.001) # 100 - 120 (0.0005)
    # # summary(Model(), input_size=[(1, 100, 100), (1, 100, 100)])
    #
    # training(net, loader, optimizer, loss, epochs=50)

    test_dataloader = DataLoader(ds, num_workers=2, batch_size=1, shuffle=True)
    dataiter = iter(test_dataloader)
    x0, _, _ = next(dataiter)
    true = 0
    false = 0
    for i in range(200):
        _, x1, label2 = next(dataiter)
        concatenated = torch.cat((x0, x1), 0)
        out1, out2 = net(x0.to(device), x1.to(device))
        dist = F.pairwise_distance(out1, out2)
        if dist.item() < 0.7:
            true += 1
        else:
            false += 1
        # utils.imshow(tv.utils.make_grid(concatenated), f'dissimilarity {dist.item():.2f}')
    print(f'accuracy: {true/100.:.2f}')


if __name__ == '__main__':
    # ds_path = r'C:\LockMe_DATA\ATNT'
    # transformation = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])
    #
    # ds = ModelDataset(ds_path=ds_path, transform=transformation)
    # ds.generate_faces()
    main()
