import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import Adam
# from tensorflow_addons.losses import ContrastiveLoss
from model_tf.model import SNN
from model_tf.dataset import generate_pairs_labels


def calculate_accuracy(y_true, y_pred):
    y_pred_binary = tf.cast(y_pred > 0.5, tf.int32)  # Convert similarity values to binary predictions (0 or 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred_binary), tf.float32))
    return accuracy


def train(model, parameters, num_epochs, batch_size):
    # Lists to store training progress
    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []

    # Train the model
    for epoch in range(num_epochs):
        history = model.model.fit(
            x=[parameters['train_pairs'][:, 0], parameters['train_pairs'][:, 1]],
            y=parameters['train_labels'],
            validation_data=([parameters['val_pairs'][:, 0], parameters['val_pairs'][:, 1]], parameters['val_labels']),
            batch_size=batch_size, epochs=1
        )

        # Get training and validation loss
        train_loss = history.history['loss'][0]
        val_loss = history.history['val_loss'][0]
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # Get training and validation accuracy
        train_pred = model.model.predict([parameters['train_pairs'][:, 0], parameters['train_pairs'][:, 1]])
        val_pred = model.model.predict([parameters['val_pairs'][:, 0], parameters['val_pairs'][:, 1]])
        train_acc = calculate_accuracy(parameters['train_labels'], train_pred)
        val_acc = calculate_accuracy(parameters['val_labels'], val_pred)
        train_accuracy_history.append(train_acc)
        val_accuracy_history.append(val_acc)

        # Print epoch results
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    model.model.save('siamese_model.h5')

    # Plot training progress
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy_history, label='Train Acc')
    plt.plot(val_accuracy_history, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    # Set the number of CPU cores to use
    num_cpu_cores = os.cpu_count() - 1  # Set the desired number of CPU cores

    # Limit TensorFlow to use the specified number of CPU cores
    tf.config.threading.set_intra_op_parallelism_threads(num_cpu_cores)
    tf.config.threading.set_inter_op_parallelism_threads(num_cpu_cores)

    # Set the input shape and create the model
    input_shape = (100, 100, 1)
    model = SNN(input_shape)

    # Compile the model with the contrastive loss
    model.model.compile(optimizer=Adam(learning_rate=0.0005), loss=SNN.contrastive_loss, metrics=['accuracy'])
    model.model.summary()

    # Set the path to the directory containing the AT&T dataset
    dataset_dir = r"C:\LockMe_DATA\ATNT"
    data_dict = generate_pairs_labels(dataset_dir, input_shape, 250, 250)

    # Split the data into training and validation sets
    train_size = int(0.8 * len(data_dict['pairs']))  # 80% for training
    train_pairs = data_dict['pairs'][:train_size]
    train_labels = data_dict['labels'][:train_size]
    val_pairs = data_dict['pairs'][train_size:]
    val_labels = data_dict['labels'][train_size:]
    parameters = {'train_pairs': train_pairs,
                  'train_labels': train_labels,
                  'val_pairs': val_pairs,
                  'val_labels': val_labels}
    train(model, parameters, 20, 32)
    model.model.save('siamese_model.h5')


if __name__ == '__main__':
    main()
