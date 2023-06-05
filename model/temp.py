import shutil
import sys
from model.SNN import Net
from dataset import ModelDataset
from model.SNN import Net
import model.config as config
import model.model_utils as utils
import torch
from facenet_pytorch import MTCNN
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
from torchsummary import summary
import cv2 as cv
import os
import matplotlib.pyplot as plt


def filter_ds2(path_to_db, new_base_path, out_path, lower=1, upper=sys.maxsize, start_ind=0):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    folders = os.listdir(path_to_db)
    if new_base_path is None:
        folders = []
        new_base_path = path_to_db
    else:
        os.makedirs(new_base_path)

    for folder in folders:
        os.mkdir(os.path.join(new_base_path, folder))

        dir_path = os.path.join(path_to_db, folder)
        if not (lower <= len(os.listdir(dir_path)) <= upper):
            continue

        for file in os.listdir(dir_path):
            old_path = os.path.join(dir_path, file)
            image = cv.imread(old_path)
            boxes, conf = mtcnn.detect(image)
            if boxes is None or len(boxes) != 1:
                continue

            old_path = os.path.join(dir_path, file)
            new_path = os.path.join(new_base_path, folder, file)
            shutil.copy(old_path, new_path)
        if lower <= len(os.listdir(os.path.join(new_base_path, folder))) <= upper:
            print(f'passed: {dir_path}')
        else:
            shutil.rmtree(os.path.join(new_base_path, folder))

    people = os.listdir(new_base_path)
    os.makedirs(out_path)
    print('filtered successfully')

    index = start_ind
    for p in people:
        new_dir = os.path.join(out_path, f's{index}')
        os.makedirs(new_dir)

        file_ind = 1
        for j, file in enumerate(os.listdir(os.path.join(new_base_path, p))):
            image = cv.imread(os.path.join(new_base_path, p, file))
            boxes, conf = mtcnn.detect(image)
            if boxes is None or len(boxes) != 1:
                continue

            boxes = boxes.astype(int)
            frame = ModelDataset.create_image(image, boxes[0])
            cv.imwrite(os.path.join(new_dir, f'{file_ind}.pgm'), frame)
            file_ind += 1
        if lower <= len(os.listdir(new_dir)) <= upper:
            print(f'subject: s{index} added')
            index += 1
        else:
            shutil.rmtree(new_dir)


def augment_ds(path):
    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    for folder in folders:
        length = len(os.listdir(os.path.join(path, folder)))

        for i, filename in enumerate(os.listdir(os.path.join(path, folder))):
            path_bright = os.path.join(path, folder, f'{1 + i + length}.pgm')
            path_dark = os.path.join(path, folder, f'{1 + i + 2 * length}.pgm')
            path_flip = os.path.join(path, folder, f'{1 + i + 3 * length}.pgm')
            current = os.path.join(path, folder, f'{1 + i}.pgm')

            frame = cv.imread(current, cv.IMREAD_GRAYSCALE)
            darkened = cv.convertScaleAbs(frame, alpha=1.1, beta=0)
            brightened = cv.convertScaleAbs(frame, alpha=0.7, beta=0)
            flipped = cv.flip(frame, 1)

            cv.imwrite(path_dark, darkened)
            cv.imwrite(path_bright, brightened)
            cv.imwrite(path_flip, flipped)
        print(f'finished subject: {folder}')


if __name__ == '__main__':
    filter_ds2(r'C:\Users\urikh\Downloads\lfw_filtered', None, r'C:\LockMe_DATA\samples', lower=5, start_ind=41)
    augment_ds(r'C:\LockMe_DATA\samples')
    filter_ds2(r'C:\Users\urikh\Downloads\lfw_filtered', None, r'C:\LockMe_DATA\my_ATNT_DS\TEST', lower=2, upper=4, start_ind=1)