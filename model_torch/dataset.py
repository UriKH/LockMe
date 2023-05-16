import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets


class ModelDataset(Dataset):
    def __init__(self, ds_path, transform=None):
        self.ds_path = ds_path
        self.dataset = datasets.ImageFolder(root=ds_path)
        self.transform = transform

    def __getitem__(self, index):
        img0_tup = random.choice(self.dataset.imgs)

        pair = random.randint(0, 1)
        if pair:
            while True:
                img1_tup = random.choice(self.dataset.imgs)
                if img0_tup[1] == img1_tup[1]:
                    break
        else:
            while True:
                img1_tup = random.choice(self.dataset.imgs)
                if img0_tup[1] != img1_tup[1]:
                    break

        img0 = Image.open(img0_tup[0])
        img1 = Image.open(img1_tup[0])

        img0 = img0.convert('L')
        img1 = img1.convert('L')

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tup[1] != img0_tup[1])], dtype=np.float32))

    def generate_faces(self):
        folders = [file for file in os.listdir(self.ds_path) if os.path.isdir(os.path.join(self.ds_path, file))]
        new_path = os.path.join(self.ds_path, f's{len(folders) + 1}')
        if not os.path.exists(new_path):
            os.mkdir(new_path)

        cap = cv.VideoCapture(0)
        cntr = 0
        while True:
            ret, frame = cap.read()
            frame = frame[150:350, 150:350]
            frame = cv.resize(frame, (500, 500))
            frame = cv.flip(frame, 1)
            cv.imshow('cam', frame)

            key = cv.waitKey(1)
            if key == ord('c'):
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                cntr += 1
                cv.imwrite(f'{os.path.join(new_path, f"{cntr}")}.pgm', frame)
                print(f'image saved - {cntr} images saved until now')
            elif key == ord('q'):
                break
        new_path = r'C:\LockMe_DATA\ATNT\s42'
        for image_path in [os.path.join(new_path, filename) for filename in os.listdir(new_path)]:
            image = cv.imread(image_path)
            plt.axis('off')
            plt.imshow(image)
            plt.show()
            print(image_path)

    def __len__(self):
        return len(self.dataset.imgs)
