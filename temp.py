import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2 as cv

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from facenet_pytorch import MTCNN
import model.dataset as ds


def filter1(path):
    folders = [os.path.join(path, directory) for directory in os.listdir(path) if os.path.isdir(os.path.join(path, directory))]
    files = []
    for folder in folders:
        for file in os.listdir(folder):
            files.append(os.path.join(folder, file))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    for file in files:
        image = cv.imread(file)
        boxes, conf = mtcnn.detect(image)
        if boxes is None or len(boxes) != 1:
            print(file)
            os.remove(file)

# filter1(r'C:\LockMe_DATA\LFW_filtered')
# ds.ModelDataset.create_samples_from_folders(r'C:\LockMe_DATA\LFW_filtered', 47)
dataset = ds.ModelDataset(r'C:\LockMe_DATA\other')
dataset.augment_dataset(r'C:\LockMe_DATA\temp', discover=False)