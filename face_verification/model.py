import cv2 as cv
import os
import random
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
import keras.layers as layers
import keras.models as models


class Prepare:
    base_data_path = r'C:\LockMe_DATA'
    positives_path = os.path.join(base_data_path, 'data', 'positive')
    negatives_path = os.path.join(base_data_path, 'data', 'negative')
    anchors_path = os.path.join(base_data_path, 'data', 'anchor')

    def __init__(self):
        os.makedirs(Prepare.positives_path)
        os.makedirs(Prepare.negatives_path)
        os.makedirs(Prepare.anchors_path)

        lfw_path = os.path.join(Prepare.base_data_path, 'lfw')

        for directory in os.listdir(lfw_path):
            for file in os.listdir(os.path.join(lfw_path, directory)):
                current_path = os.path.join(lfw_path, directory, file)
                new_path = os.path.join(Prepare.negatives_path, file)
                os.replace(current_path, new_path)

    @staticmethod
    def prepare_gpus():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


class L1Dist(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_e, validation_e):
        # comparing similarity when calling the layer
        return tf.math.abs(input_e - validation_e) #(this is L1Dist)
        # return tf.norm(input_e - validation_e)


class Model:
    def __init__(self):
        self.embedding = Model.embedding_model()
        self.model = self.model()
        self.model.summary()

    @staticmethod
    def embedding_model():
        x = layers.Input(shape=(105, 105, 3), name='input_image')
        conv1 = layers.Conv2D(64, (10, 10), activation='relu')(x)
        mp1 = layers.MaxPool2D(64, (2, 2), padding='same')(conv1)

        conv2 = layers.Conv2D(128, (7, 7), activation='relu')(mp1)
        mp2 = layers.MaxPool2D(64, (2, 2), padding='same')(conv2)

        conv3 = layers.Conv2D(128, (4, 4), activation='relu')(mp2)
        mp3 = layers.MaxPool2D(64, (2, 2), padding='same')(conv3)

        conv4 = layers.Conv2D(256, (4, 4), activation='relu')(mp3)
        fc = layers.Flatten()(conv4)
        d = layers.Dense(512, activation='sigmoid')(fc)     # originally 4096 (much less expensive with 512)

        return models.Model(inputs=[x], outputs=[d], name='embedding')

    def model(self):
        anchor = layers.Input(name='anchor_img', shape=(105, 105, 3))
        validation = layers.Input(name='validation_img', shape=(105, 105, 3))

        dist_l = L1Dist()
        dist_l._name = 'distance'

        distances = dist_l(self.embedding(anchor), self.embedding(validation))

        classifier = layers.Dense(1, activation='sigmoid')(distances)

        return models.Model(inputs=[anchor, validation], outputs=classifier, name='SiameseNet')

Model()
