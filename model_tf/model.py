import math

import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, MaxPool2D, Dropout, GlobalAveragePooling2D


class SNN:
    def __init__(self, input_shape):
        self.input_shape = input_shape

        # Base convolutional neural network (CNN) for embedding generation
        self.base_cnn = tf.keras.Sequential([
            Conv2D(96, kernel_size=11, strides=4, activation='leaky_relu', kernel_initializer='glorot_uniform'),
            MaxPooling2D(pool_size=3, strides=2),
            Dropout(0.001),

            Conv2D(256, kernel_size=5, strides=1, activation='leaky_relu', kernel_initializer='glorot_uniform'),
            MaxPooling2D(pool_size=2, strides=2),
            Dropout(0.001),

            Conv2D(384, kernel_size=3, strides=1, activation='leaky_relu', kernel_initializer='glorot_uniform'),
            Flatten(),
            Dense(1024, activation='leaky_relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dense(256, activation='leaky_relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dense(512)


            # Conv2D(64, (10, 10), activation='relu'),
            # MaxPool2D(64, (2, 2), padding='same'),
            #
            # Conv2D(128, (7, 7), activation='relu'),
            # MaxPool2D(64, (2, 2), padding='same'),
            #
            # Conv2D(128, (4, 4), activation='relu'),
            # MaxPool2D(64, (2, 2), padding='same'),
            #
            # Conv2D(256, (4, 4), activation='relu'),
            # Flatten(),
            # Dense(512, activation='sigmoid')
        ])

        self.model = self.create_siamese_model()

    @staticmethod
    def distance_layer():
        return Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]), axis=-1, keepdims=True)))

    @staticmethod
    def calc_similarity(encodings: [list, list]):
        # Define the similarity measure (Euclidean distance)
        distance = SNN.distance_layer()

        # Connect the encoded tensors to the similarity measure
        # similarity = 1. / (1 + math.e ** -distance(encodings))
        similarity = distance(encodings)
        return similarity

    def create_siamese_model(self):
        # Define the input tensors for image 1 and image 2
        input_image_1 = Input(shape=self.input_shape)
        input_image_2 = Input(shape=self.input_shape)

        # Connect the input tensors to the base CNN
        encoded_image_1 = self.base_cnn(input_image_1)
        encoded_image_2 = self.base_cnn(input_image_2)

        similarity = SNN.calc_similarity([encoded_image_1, encoded_image_2])
        siamese_model = Model(inputs=[input_image_1, input_image_2], outputs=similarity)
        return siamese_model

    @staticmethod
    def contrastive_loss(y_true, y_pred):
        margin = 1.
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_mean(y_true * tf.square(y_pred) +
                              (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0.)))
