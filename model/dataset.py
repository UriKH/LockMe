import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2 as cv

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.datasets as datasets
from facenet_pytorch import MTCNN

import model.config as config
import model.model_utils as utils


class ModelDataset(Dataset):
    """
    The dataset class for the model
    """

    def __init__(self, root, transform=None, train_ratio=0.8):
        self.ds_path = root
        self.dataset = datasets.ImageFolder(root=root)
        self.transform = transform
        self.train_ratio = train_ratio

    def __getitem__(self, index):
        """
        Get a dataset pair with a label for training
        :return: image1, image2, label (0 if of same subject, else 1)
        """
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

    def __len__(self):
        return len(self.dataset.imgs)

    def split_dataset(self):
        # Compute the number of samples for train/validation split
        num_samples = len(self.dataset)
        num_train = int(self.train_ratio * num_samples)
        num_valid = num_samples - num_train

        # Perform the train/validation split
        train_dataset, valid_dataset = random_split(self, [num_train, num_valid])
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=utils.get_workers())
        valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=utils.get_workers())
        return train_loader, valid_loader


    @staticmethod
    def create_image(image, box):
        """
        Create an image in a usable format for the model (preprocess)
        :param image: the original image to cut from
        :param box: face bounding box in format top left bottom right
        :return: the face on white background
        """
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

    @staticmethod
    def create_samples_from_folders(parent=r'C:\LockMe_DATA\temp', start=0):
        """
        Create a basic dataset from folders of images in form of the AT&T dataset
        :param parent: the parent directory of all subjects
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device
        )

        people = os.listdir(parent)
        for i, p in enumerate(people):
            new_dir = os.path.join(parent, f's{i + 1 + start}')
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
                cv.imwrite(os.path.join(new_dir, f'{j + 1}.pgm'), frame)

    @staticmethod
    def filter_lfw(path_to_lfw, new_base_path, threshold=10):
        """
        Filter the LFW database for subjects with more images that the threshold
        :param path_to_lfw: the path to the current LFW folder
        :param new_base_path: path to the filtered dataset
        :param threshold: the amount of samples to filter on
        """
        folders = os.listdir(path_to_lfw)
        for folder in folders:
            dir_path = os.path.join(path_to_lfw, folder)
            if len(os.listdir(dir_path)) >= threshold:
                os.mkdir(os.path.join(new_base_path, folder))
                for file in os.listdir(dir_path):
                    old_path = os.path.join(dir_path, file)
                    new_path = os.path.join(new_base_path, folder, file)
                    shutil.copy(old_path, new_path)

    @staticmethod
    def filter_faces(path):
        """
        Remove all images which not contain 1 face exactly
        :param path: path to the dataset parent folder
        """
        folders = [os.path.join(path, directory) for directory in os.listdir(path) if
                   os.path.isdir(os.path.join(path, directory))]
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

    def cam_face_generation(self, path=None):
        """
        Generate face from the camera
        """

        if path is None:
            folders = [file for file in os.listdir(self.ds_path) if os.path.isdir(os.path.join(self.ds_path, file))]
            new_path = os.path.join(self.ds_path, f's{len(folders) + 1}')
            if not os.path.exists(new_path):
                os.mkdir(new_path)
        else:
            new_path = path

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

    def augment_dataset(self, new_ds_path, discover=True):
        """
        Transfer the current dataset to another folder and augment the images
        :param new_ds_path: path to the new dataset folder
        :param discover: discover faces from raw images
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device
        )
        cntr = 0

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
