import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras.layers as layers
import keras.models as models

import io
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

faces = fetch_lfw_people(min_faces_per_person=60, color=True)
X = faces.images
y = faces.target
target_names = faces.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train / 255.0
X_test = X_test / 255.0

model = tf.keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=10, activation='relu', input_shape=(62, 47, 3)),
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
