import os
import cv2 as cv
import numpy as np
import random


def create_data(dataset_dir, input_shape, data_samples=480):
    pairs_path = 'image_pairs.npy'
    labels_path = 'labels.npy'

    if os.path.exists(os.path.join(os.getcwd(), pairs_path)) and os.path.exists(os.path.join(os.getcwd(), labels_path)):
        print('data found in files')
        image_pairs = np.load(pairs_path)
        labels = np.load(labels_path)
        return {'pairs': image_pairs, 'labels': labels}

    # Set the number of individuals and the number of images per individual in the dataset
    num_individuals = 40
    num_images_per_individual = 10

    # Create empty lists to store image pairs and corresponding labels
    image_pairs = []
    labels = []

    # Iterate over individuals and create image pairs
    for i in range(1, num_individuals + 1):
        # Create a list to store images of the current individual
        images = []

        # Iterate over images for the current individual
        for j in range(1, num_images_per_individual + 1):
            # Set the path to the current image
            image_path = os.path.join(dataset_dir, f"s{i}", f"{j}.pgm")

            # Read the image using OpenCV
            image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            if image is None:
                print(f'could not read file: {image_path}')
                continue

            image = cv.resize(image, (input_shape[1], input_shape[0]))
            # Normalize the pixel values between 0 and 1
            image = image.astype(np.float32) / 255.0

            # Append the image to the list
            images.append(image)

        # Create positive pairs (images of the same individual)
        for k in range(num_images_per_individual - 1):
            for l in range(k + 1, num_images_per_individual):
                image_pairs.append([images[k], images[l]])
                labels.append(1)

        # Create negative pairs (images of different individuals)
        for m in range(num_individuals - 1):
            for n in range(m + 1, num_individuals):
                for p in range(num_images_per_individual):
                    image_pairs.append([images[p], images[(p + m) % num_images_per_individual]])
                    labels.append(0)

    # Convert the image pairs and labels to NumPy arrays
    image_pairs = np.array(image_pairs)
    labels = np.array(labels)

    # Shuffle the data randomly
    indices = np.random.permutation(len(image_pairs))
    image_pairs = image_pairs[indices]
    labels = labels[indices]

    print('data loaded')

    # Save the image pairs and labels data
    np.save(pairs_path, image_pairs)
    np.save(labels_path, labels)

    print(f'data saved in external file ({pairs_path}, {labels_path})')
    return {'pairs': image_pairs, 'labels': labels}


def generate_pairs_labels(dataset_folder, input_shape, num_positive_pairs, num_negative_pairs):
    image_paths = []
    labels = []

    # Load the images and labels from the dataset
    for subject in range(1, 41):
        subject_folder = os.path.join(dataset_folder, 's' + str(subject))
        subject_images = os.listdir(subject_folder)

        for image_name in subject_images:
            image_path = os.path.join(subject_folder, image_name)
            image_paths.append(image_path)
            labels.append(subject - 1)  # Use subject index as label (0-39)

    image_pairs = []
    pair_labels = []

    # Generate positive pairs
    for _ in range(num_positive_pairs):
        subject = random.randint(1, 40)
        subject_folder = os.path.join(dataset_folder, 's' + str(subject))
        subject_images = os.listdir(subject_folder)

        image1 = random.choice(subject_images)
        image2 = random.choice(subject_images)

        image_path1 = os.path.join(subject_folder, image1)
        image_path2 = os.path.join(subject_folder, image2)

        img1 = cv.imread(image_path1, cv.IMREAD_GRAYSCALE) / 255.
        img2 = cv.imread(image_path2, cv.IMREAD_GRAYSCALE) / 255.
        img1 = cv.resize(img1, (input_shape[1], input_shape[0]))
        img2 = cv.resize(img2, (input_shape[1], input_shape[0]))
        image_pairs.append([img1, img2])
        pair_labels.append(1.)  # Positive pair label

    # Generate negative pairs
    for _ in range(num_negative_pairs):
        subject1 = random.randint(1, 40)
        subject2 = random.randint(1, 40)

        # Make sure subject2 is different from subject1
        while subject2 == subject1:
            subject2 = random.randint(1, 40)

        subject1_folder = os.path.join(dataset_folder, 's' + str(subject1))
        subject2_folder = os.path.join(dataset_folder, 's' + str(subject2))

        subject1_images = os.listdir(subject1_folder)
        subject2_images = os.listdir(subject2_folder)

        image1 = random.choice(subject1_images)
        image2 = random.choice(subject2_images)

        image_path1 = os.path.join(subject1_folder, image1)
        image_path2 = os.path.join(subject2_folder, image2)

        img1 = cv.imread(image_path1, cv.IMREAD_GRAYSCALE) / 255.
        img2 = cv.imread(image_path2, cv.IMREAD_GRAYSCALE) / 255.
        img1 = cv.resize(img1, (input_shape[1], input_shape[0]))
        img2 = cv.resize(img2, (input_shape[1], input_shape[0]))
        image_pairs.append([img1, img2])
        pair_labels.append(0.)  # Negative pair label

    return {'pairs': np.array(image_pairs), 'labels': np.array(pair_labels)}