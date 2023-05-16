import tensorflow as tf
from keras import layers


class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def call(self, out1, out2, label):
        label = tf.cast(label, out2.dtype)
        euclidean_dist = tf.norm(out1 - out2, axis=1, keepdims=True)
        loss = tf.reduce_mean((1 - label) * tf.square(euclidean_dist) +
                              label * tf.square(tf.maximum(self.margin - euclidean_dist, 0.0)))
        return loss


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        # Convolutional blocks
        self.cnn = tf.keras.Sequential([
            layers.Conv2D(96, kernel_size=11, strides=4, activation='relu', input_shape=(100, 100, 1)),
            layers.MaxPooling2D(pool_size=3, strides=2),

            layers.Conv2D(256, kernel_size=5, strides=1, activation='relu'),
            layers.MaxPooling2D(pool_size=2, strides=2),

            layers.Conv2D(384, kernel_size=3, strides=1, activation='relu')
        ])

        # Fully connected part for embedding generation
        self.fc = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(2)
        ])

    def call(self, input1, input2):
        out1 = self.cnn(input1)
        out1 = self.fc(out1)

        out2 = self.cnn(input2)
        out2 = self.fc(out2)

        return out1, out2
