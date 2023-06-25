import shutil
import os
import sys
import cv2 as cv


def convert_to_format(folder_path, format):
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        new_path = '.'.join(os.path.join(folder_path, filename).split('.')[:-1]) + f'.{format}'
        cv.imwrite(new_path, img)
        os.remove(path)


def convert_parent(folder_path, format):
    for folder in os.listdir(folder_path):
        convert_to_format(os.path.join(folder_path, folder), format)


def main():
   convert_to_format(r'G:\My Drive\datasets\celeba\s4', 'png')


if __name__ == '__main__':
    main()