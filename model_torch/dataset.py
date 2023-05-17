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
from facenet_pytorch import MTCNN


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

    @staticmethod
    def create_image(image, box):
        x1, y1, x2, y2 = box

        h, w, _ = image.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        width = x2 - x1
        height = y2 - y1
        cut = image[y1: y2, x1: x2]
        prop = width / height

        if width > height:
            width = 300
            height = int(300 / prop)
        else:
            width = int(prop * 300)
            height = 300
        cut = cv.resize(cut, (width, height))

        canvas = np.ones((300, 300, 3), dtype=np.uint8) * 255
        start_x = (300 - width) // 2
        start_y = (300 - height) // 2
        canvas[start_y: start_y + height, start_x: start_x + width] = cut

        canvas = cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
        return canvas

    def generate_faces(self):
        folders = [file for file in os.listdir(self.ds_path) if os.path.isdir(os.path.join(self.ds_path, file))]

        new_path = r'C:\LockMe_DATA\new_dataYAY'
        # new_path = os.path.join(self.ds_path, f's{len(folders) + 1}')
        if not os.path.exists(new_path):
            os.mkdir(new_path)

        cap = cv.VideoCapture(0)
        cntr = 0
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device
        )

        while True:
            ret, frame = cap.read()
            frame = frame[150:350, 150:350]

            frame = cv.resize(frame, (500, 500))
            frame = cv.flip(frame, 1)
            cv.imshow('cam', frame)

            key = cv.waitKey(1)
            if key == ord('c'):
                boxes, conf = mtcnn.detect(frame)
                boxes = boxes.astype(int)
                boxes = [box for i, box in enumerate(boxes) if conf[i] >= 0.95]

                frame = ModelDataset.create_image(frame, boxes[0])

                cntr += 1
                cv.imwrite(f'{os.path.join(new_path, f"{cntr}")}.pgm', frame)
                print(f'image saved - {cntr} images saved until now')
            elif key == ord('q'):
                break

        for image_path in [os.path.join(new_path, filename) for filename in os.listdir(new_path)]:
            image = cv.imread(image_path)
            plt.axis('off')
            plt.imshow(image)
            plt.show()
            print(image_path)

    def transfer_dataset(self, discover=True):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device
        )
        cntr = 0
        new_ds_path = r'C:\LockMe_DATA\augmented'

        folders = [name for name in os.listdir(self.ds_path) if os.path.isdir(os.path.join(self.ds_path, name))]
        for folder in folders:
            if not os.path.exists(os.path.join(new_ds_path, folder)):
                os.makedirs(os.path.join(new_ds_path, folder))
            for i in range(1, len(os.listdir(os.path.join(self.ds_path, folder))) + 1):
                length = len(os.listdir(os.path.join(self.ds_path, folder)))
                old_path = os.path.join(self.ds_path, folder, f'{i}.pgm')
                new_path = os.path.join(new_ds_path, folder, f'{i}.pgm')
                new_path_bright = os.path.join(new_ds_path, folder, f'{i + length}.pgm')
                new_path_dark = os.path.join(new_ds_path, folder, f'{i + 2 * length}.pgm')
                new_path_flip = os.path.join(new_ds_path, folder, f'{i + 3 * length}.pgm')

                frame = cv.imread(old_path, cv.IMREAD_GRAYSCALE)
                if discover:
                    frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

                    frame = cv.resize(frame, (500, 500))
                    boxes, conf = mtcnn.detect(frame)
                    boxes = boxes.astype(int)
                    boxes = [box for i, box in enumerate(boxes) if conf[i] >= 0.95]

                darkened = cv.convertScaleAbs(frame, alpha=1.1, beta=0)
                brightened = cv.convertScaleAbs(frame, alpha=0.7, beta=0)

                if discover:
                    frame = ModelDataset.create_image(frame, boxes[0])
                    darkened = ModelDataset.create_image(darkened, boxes[0])
                    brightened = ModelDataset.create_image(brightened, boxes[0])
                flipped = cv.flip(frame, 1)
                try:
                    cntr += 1
                    cv.imwrite(new_path, frame)
                    cv.imwrite(new_path_dark, darkened)
                    cv.imwrite(new_path_bright, brightened)
                    cv.imwrite(new_path_flip, flipped)
                except:
                    print(f'error in writing files: {new_path} or the others')
                print(f'image saved - {cntr} images saved until now')

    def __len__(self):
        return len(self.dataset.imgs)
