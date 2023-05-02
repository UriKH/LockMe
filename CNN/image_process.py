import tensorflow as tf
import cv2 as cv
import face_recognition as fr
import numpy as np
import initialize as init
import math


class Image:
    model = init.Init.facenet_model
    buffer = 10

    def __init__(self, image):
        self.image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.embeddings_dict = {}

        self.create_embeddings()

    @staticmethod
    def preprocess(image, shape):
        new_img = image.copy()
        new_img = cv.resize(new_img, shape)
        return new_img

    @staticmethod
    def xyxy_to_xywh(top, right, bottom, left):
        return left, top, right - left, bottom - top

    @staticmethod
    def get_embedding(face):
        face = Image.preprocess(face, (160, 160))
        embedding = Image.model.embeddings([face])
        return embedding

    def __get_coords(self):
        coords = fr.face_locations(self.image)
        coords = [Image.xyxy_to_xywh(coord[0], coord[1], coord[2], coord[3]) for coord in coords]
        return coords

    def __extract_faces(self, coords):
        height, width, _ = self.image.shape

        faces = list()
        for x, y, w, h in coords:
            x -= Image.buffer if x - Image.buffer >= 0 else 0
            y -= Image.buffer if y - Image.buffer >= 0 else 0
            w = w + 2 * Image.buffer if x + w + 2 * Image.buffer < width else width
            h = h + 2 * Image.buffer if y + h + 2 * Image.buffer < height else height

            face_crop = self.image[y: y + h, x: x + w]
            if w == h:
                faces.append(face_crop)
                continue

            noise_shape = (h, h, 3) if h > w else (w, w, 3)
            noise = tf.random.uniform(shape=noise_shape, minval=0, maxval=255, dtype=tf.int32)
            noise_img = noise.numpy().astype(np.uint8)

            if h > w:
                center = math.floor((h - w) / 2)
                noise_img[:, center: center + w, :] = face_crop
            else:
                center = math.floor((w - h) / 2)
                noise_img[center: center + h, :, :] = face_crop

            faces.append(noise_img)
        return faces

    def create_embeddings(self):
        for coord in self.__get_coords():
            self.embeddings_dict[coord] = None
        if len(self.embeddings_dict.keys()) == 0:
            return

        for coord, face in zip(self.embeddings_dict.keys(), self.__extract_faces(self.embeddings_dict.keys())):
            self.embeddings_dict[coord] = Image.get_embedding(face)


if __name__ == '__main__':
    image = cv.imread(r"C:\Users\urikh\OneDrive\pictures\Camera Roll\WIN_20200315_10_49_03_Pro.jpg")
    image_obj = Image(image)
    print('finished')
