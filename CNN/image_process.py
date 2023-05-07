import tensorflow as tf
import cv2 as cv
import numpy as np
import math
import torch
from PIL import Image as PImage

from CNN.initialize import Init
from logger import Logger
from messages import Messages as msg


class Image(Init):
    buffer = 10
    conf_thresh = 0.95

    def __init__(self, image):
        if Init.facenet_model is None:
            super().__init__()
        self.image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.embeddings_dict = {}

        self.create_embeddings()
        self.image = cv.cvtColor(self.image, cv.COLOR_RGB2BGR)
        self.x_pos, self.y_pos = None, None

    # @staticmethod
    # def preprocess(image, shape):
    #     new_img = image.copy()
    #     new_img = cv.resize(new_img, shape)
    #     new_img = np.around(np.array(new_img) / 255.0, decimals=12)
    #     return new_img

    # @staticmethod
    # @Logger(msg.Info.embeddings_generated, Logger.info).time_it
    # def get_embeddings(faces):
    #     faces = [Image.preprocess(face, (160, 160)) for face in faces]
    #     embeddings = Init.facenet_model.embeddings(faces)
    #     for i, embedding in enumerate(embeddings):
    #         embeddings[i] = embedding / np.linalg.norm(embedding, ord=2)
    #
    #     Logger(msg.Info.embeddings_generated, level=Logger.info)
    #     return embeddings

    @Logger(msg.Info.faces_located, Logger.info).time_it
    def __get_coords(self):
        # faces_data = Init.facenet_model.extract(self.image)
        # coords = [tuple(face['box']) for face in faces_data if face['confidence'] > Image.conf_thresh]
        # return coords
        boxes, conf = Init.mtcnn.detect(self.image)
        boxes = boxes.astype(int)
        boxes = [box for i, box in enumerate(boxes) if conf[i] >= Image.conf_thresh]
        return boxes

    # def __extract_faces(self, coords):
    #         height, width, _ = self.image.shape
    #
    #         faces = list()
    #         for x, y, w, h in coords:
    #             x -= Image.buffer if x - Image.buffer >= 0 else 0
    #             y -= Image.buffer if y - Image.buffer >= 0 else 0
    #             w = w + 2 * Image.buffer if x + w + 2 * Image.buffer < width else width - x
    #             h = h + 2 * Image.buffer if y + h + 2 * Image.buffer < height else height - y
    #
    #             face_crop = self.image[y: y + h, x: x + w]
    #             if w == h:
    #                 faces.append(face_crop)
    #                 continue
    #
    #             noise_shape = (h, h, 3) if h > w else (w, w, 3)
    #             noise = tf.random.uniform(shape=noise_shape, minval=0, maxval=255, dtype=tf.int32)
    #             noise_img = noise.numpy().astype(np.uint8)
    #
    #             if h > w:
    #                 center = math.floor((h - w) / 2)
    #                 noise_img[:, center: center + w, :] = face_crop
    #             else:
    #                 center = math.floor((w - h) / 2)
    #                 noise_img[center: center + h, :, :] = face_crop
    #
    #             faces.append(noise_img)
    #         return faces

    @staticmethod
    def xyxy_to_xywh(box):
        return box[0], box[1], box[2] - box[0], box[3] - box[1]

    @Logger(msg.Info.embeddings_generated, Logger.info).time_it
    def create_embeddings(self):
        faces = []
        for box in self.__get_coords():
            x1, y1, x2, y2 = box
            temp_img = PImage.fromarray(self.image[y1: y2, x1: x2])
            x_aligned = Init.mtcnn(temp_img)
            if x_aligned is None:
                continue
            faces.append(x_aligned)
            self.embeddings_dict[Image.xyxy_to_xywh(box)] = None
        if len(self.embeddings_dict.keys()) == 0:
            return

        aligned = torch.stack(faces).to(Init.device)
        embeddings = Init.resnet(aligned).detach().cpu()
        for box, embedding in zip(self.embeddings_dict.keys(), embeddings):
            self.embeddings_dict[box] = embedding

        # for coord in self.__get_coords():
        #     self.embeddings_dict[coord] = None
        # if len(self.embeddings_dict.keys()) == 0:
        #     return

        # faces = self.__extract_faces(self.embeddings_dict.keys())
        # for coord, embedding in zip(self.embeddings_dict.keys(), Image.get_embeddings(faces)):
        #     self.embeddings_dict[coord] = embedding

    def choose_face(self):
        from camera_runner import Camera

        def mouse_callback(event, x, y, flags, params):
            if event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
                self.x_pos = x
                self.y_pos = y
                Logger(f'mouse clicked ({x}, {y})', Logger.info).log()

        cv.setMouseCallback(Camera.window_name, mouse_callback)

        new_img = self.image
        for i, (x, y, w, h) in enumerate(self.embeddings_dict.keys()):
            cv.rectangle(new_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv.rectangle(new_img, (x, y + h), (x + w, y + h + 35), (0, 0, 255), cv.FILLED)
            cv.putText(new_img, str(i + 1), (x + 7, y + h + 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        Logger(msg.Requests.face_index, Logger.message).log()
        while True:
            cv.imshow(Camera.window_name, new_img)
            cv.waitKey(1)
            if self.x_pos is None and self.y_pos is None:
                continue
            for i, (x, y, w, h) in enumerate(self.embeddings_dict.keys()):
                if x <= self.x_pos <= x + w and y <= self.y_pos <= y + h:
                    return list(self.embeddings_dict.values())[i]
            self.x_pos, self.y_pos = None, None
            Logger(msg.Errors.no_face_click, Logger.warning).log()


if __name__ == '__main__':
    image = cv.imread(r"C:\Users\urikh\OneDrive\pictures\Camera Roll\WIN_20200315_10_49_03_Pro.jpg")
    image_obj = Image(image)
    print('finish')
