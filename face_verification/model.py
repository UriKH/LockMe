import cv2 as cv
import os
import random
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

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


class Model:
    class L1Dist(layers.Layer):
        def __init__(self, **kwargs):
            super().__init__()

        def call(self, input_e, validation_e):
            # comparing similarity when calling the layer
            return tf.math.abs(input_e - validation_e)  # (this is L1Dist)
            # return tf.norm(input_e - validation_e)

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

        dist_l = Model.L1Dist()
        dist_l._name = 'distance'

        distances = dist_l(self.embedding(anchor), self.embedding(validation))

        classifier = layers.Dense(1, activation='sigmoid')(distances)

        return models.Model(inputs=[anchor, validation], outputs=classifier, name='SiameseNet')


def run():
    import io
    import tensorflow_addons as tfa
    import tensorflow_datasets as tfds

    train_dataset, test_dataset = tfds.load(name="lfw", split=['train[:20%]', 'train[20%:30%]'], as_supervised=True)
    dataset_full = tfds.load(name="lfw", split=['train'], as_supervised=True)[0]

    # Get the unique labels
    unique_labels = dataset_full.map(lambda image, label: label).unique()

    # Count the number of unique classes
    num_classes = len(list(unique_labels.as_numpy_iterator()))

    def _normalize_img(img, label):
        img = tf.cast(img, tf.float32) / 255.
        return img, label

    def _one_hot(text, label):
        one_hot_label = tf.one_hot(tf.strings.to_number(label, out_type=tf.int32), num_classes)
        return text, one_hot_label

    # Build your input pipelines
    train_dataset = train_dataset.shuffle(1024).batch(32)
    train_dataset = train_dataset.map(_normalize_img)
    train_dataset = train_dataset.map(_one_hot)

    test_dataset = test_dataset.batch(32)
    # data = data.batch(10)
    test_dataset = test_dataset.map(_normalize_img).map(lambda x, y: (x, tf.one_hot(y, depth=3)))

    model = tf.keras.Sequential([
        layers.Conv2D(filters=64, kernel_size=10, activation='relu', input_shape=(250, 250, 3)),
        layers.MaxPool2D(pool_size=2, padding='same'),
        layers.Dropout(0.2),

        layers.Conv2D(filters=128, kernel_size=7, activation='relu'),
        layers.MaxPool2D(pool_size=2, padding='same'),
        layers.Dropout(0.2),

        layers.Conv2D(filters=128, kernel_size=4, activation='relu'),
        layers.MaxPool2D(pool_size=2, padding='same'),
        layers.Dropout(0.2),

        layers.Conv2D(filters=256, kernel_size=4, activation='relu'),
        layers.Flatten(),
        layers.Dense(512, activation=None),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ])

    # Compile the model_torch
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tfa.losses.TripletSemiHardLoss(), metrics=['accuracy'])

    # Train the network
    history = model.fit(train_dataset, epochs=5)

    # test_loss, test_accuracy = model_torch.evaluate(test_dataset)
    # print("Test accuracy:", test_accuracy)


if __name__ == '__main__':
    run()
