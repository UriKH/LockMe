import os
import cv2 as cv
import numpy as np

from initialize import Init
from image_process import Image


class Dataset:
    every = 1. / 24. * 6   # seconds
    conf_thresh = 0.95
    parent_folder = fr'{os.getcwd()}\datasets'

    def __init__(self, folder_path):
        Init()
        self.folder_path = folder_path
        self.embeddings_dict = {}

        self.build_dataset()
        self.ds_path_x = None
        self.ds_path_y = None
        self.write_datasets_to_disk()
        del Init.database

    def build_dataset(self):
        self.folder_path = rf'{os.getcwd()}\{self.folder_path}'
        if not os.path.isdir(self.folder_path):
            print(f'folder {self.folder_path} does not exists')

        # base_folder = '\\'.join(self.folder_path.split('\\')[:-1]) + '\\'
        classes = [fr'{self.folder_path}\{f_name}' for f_name in os.listdir(self.folder_path)
                   if os.path.isdir(fr'{self.folder_path}\{f_name}')]

        # create folder structure
        paths = {}
        for cls in classes:
            paths[cls] = [path for path in os.listdir(cls) if os.path.isfile(cls + '\\' + path)]

        self.embeddings_dict = {k: [] for k in classes}
        for cls, videos in paths.items():
            for vid_name in videos:
                # for each video extract embeddings from frames
                cap = cv.VideoCapture(fr'{cls}\{vid_name}')

                total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv.CAP_PROP_FPS))
                jump = round(fps * Dataset.every)   # jump frames because nearby frames are too similar

                for i in range(0, total_frames, jump):
                    cap.set(cv.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if not ret:
                        print(fr'could not read frame {i} in video {cls}\{vid_name}...')
                        continue
                    image_data = Image(frame)
                    for embedding in image_data.embeddings_dict.values():
                        self.embeddings_dict[cls].append(embedding.numpy())
                cap.release()

    def write_datasets_to_disk(self, train=True):
        """
        Create dataset files
        :param train: True if train set else test set
        """
        x_set, y_set, classes = [], [], []
        for i, cls in enumerate(self.embeddings_dict.keys()):
            y_set += [i] * len(self.embeddings_dict[cls])
            x_set += self.embeddings_dict[cls]
            classes.append([(cls.split('\\')[-1], i)])

        y_set = np.array(y_set)
        y_set = np.expand_dims(y_set, axis=0)

        if not os.path.exists(Dataset.parent_folder):
            os.mkdir(Dataset.parent_folder)
        np.save(f'{Dataset.parent_folder}\\classes.npy', np.array(classes))

        if train:
            self.ds_path_x = f'{Dataset.parent_folder}\\x_train.npy'
            self.ds_path_y = f'{Dataset.parent_folder}\\y_train.npy'
        else:
            self.ds_path_x = f'{Dataset.parent_folder}\\x_test.npy'
            self.ds_path_y = f'{Dataset.parent_folder}\\y_test.npy'
        np.save(self.ds_path_x, np.array(x_set))
        np.save(self.ds_path_y, y_set)


def run():
    train = Dataset(r'data\train')
    # test = Dataset('path to testing folder')

    x_train = np.load(train.ds_path_x, allow_pickle=True)
    y_train = np.load(train.ds_path_y, allow_pickle=True)
    # x_test = np.load(test.ds_path_x)
    # y_test = np.load(test.ds_path_y)

    print("number of training examples = " + str(x_train.shape[0]))
    # print("number of test examples = " + str(x_test.shape[0]))
    print("X_train shape: " + str(x_train.shape))
    print("Y_train shape: " + str(y_train.shape))
    # print("X_test shape: " + str(x_test.shape))
    # print("Y_test shape: " + str(y_test.shape))


if __name__ == '__main__':
    run()
