import tensorflow as tf
import os
import utils
from temp.model import Model,  ContrastiveLoss
from temp.dataset import generate_pairs_labels
import tensorflow as tf
from keras.optimizers import Adam


def training(model, train_pairs, train_labels, val_pairs, val_labels, epochs=10, batch_size=64):
    optimizer = Adam(learning_rate=0.00025)
    loss_obj = ContrastiveLoss()
    accuracy = tf.keras.metrics.BinaryAccuracy()

    train_dataset = tf.data.Dataset.from_tensor_slices((train_pairs, train_labels)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_pairs, val_labels)).batch(batch_size)

    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            out1, out2 = model(inputs[0], inputs[1])
            loss = loss_obj.call(out1, out2, labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @tf.function
    def val_step(inputs, labels):
        out1, out2 = model(inputs[0], inputs[1])
        loss = loss_obj.call(out1, out2, labels)
        return loss

    cntr = []
    loss_hist = []
    iteration = 0
    for epoch in range(epochs):
        train_loss = tf.keras.metrics.Mean()
        val_loss = tf.keras.metrics.Mean()

        for inputs, labels in train_dataset:
            loss = train_step(inputs, labels)
            train_loss(loss)

        for inputs, labels in val_dataset:
            loss = val_step(inputs, labels)
            val_loss(loss)

        print(
            f'Epoch {epoch + 1}: Train Loss={train_loss.result():.4f}, Val Loss={val_loss.result():.4f}')

    utils.show_plot(cntr, loss_hist)
    # Save the model parameters to a file
    model.save_weights('model_params_tf')


if __name__ == '__main__':
    # Set the number of CPU cores to use
    num_cpu_cores = os.cpu_count() - 1  # Set the desired number of CPU cores

    # Limit TensorFlow to use the specified number of CPU cores
    tf.config.threading.set_intra_op_parallelism_threads(num_cpu_cores)
    tf.config.threading.set_inter_op_parallelism_threads(num_cpu_cores)

    # Set the input shape and create the model
    input_shape = (100, 100, 1)

    # Load the data
    dataset_dir = r"C:\LockMe_DATA\ATNT"
    data_dict = generate_pairs_labels(dataset_dir, input_shape, 250, 250)
    train_size = int(0.8 * len(data_dict['pairs']))  # 80% for training
    data_dict = generate_pairs_labels(dataset_dir, input_shape, 250, 250)
    train_pairs = data_dict['pairs'][:train_size]
    train_labels = data_dict['labels'][:train_size]
    val_pairs = data_dict['pairs'][train_size:]
    val_labels = data_dict['labels'][train_size:]

    model = Model()
    # Train the model
    training(model, train_pairs, train_labels, val_pairs, val_labels, epochs=50, batch_size=64)

