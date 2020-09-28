import cv2
import os, glob

import numpy as np
import tensorflow as tf
import pandas as pd

import seaborn as sns
from imutils import paths
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet101, Xception
from tensorflow.keras.layers import Input, Dense, Flatten, AveragePooling2D, Dropout, BatchNormalization, Conv2D
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

path = "C://Users//giorgos//Desktop//train//"
pathcsv = "C:/Users//giorgos//Desktop//576013_1042828_compressed_COVID-19 Radiography Database (2)//"

gpus = tf.config.experimental.list_physical_devices('GPU')
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

labels_csv = pd.read_csv(pathcsv + "mergednew.csv")
# print(labels_csv.describe())
print(labels_csv['FILE NAME'].head(1500))

filenames = [path + fname + ' .png' for fname in labels_csv["FILE NAME"]]
print(filenames[:10])

if len(os.listdir(path)) == len(filenames):
    print("Filenames match actual amount of files!")
else:
    print("Filenames do not match actual amount of files, check the target directory.")

# ftiaxnw ta labels
labels = labels_csv["target"].to_numpy()  # convert labels column to NumPy array

# See if number of labels matches the number of filenames
if len(labels) == len(filenames):
    print("Number of labels matches number of filenames!")
else:
    print("Number of labels does not match number of filenames, check data directories.")

# Find the unique label values
unique_labels = np.unique(labels)
print(len(unique_labels))

# Example: Turn one label into an array of booleans
print(labels[1500])
print(labels[0] == unique_labels)  # use comparison operator to create boolean array
# Turn every label into a boolean array
boolean_labels = [label == np.array(unique_labels) for label in labels]
print(boolean_labels[:2])

# Example: Turning a boolean array into integers
print(labels[0])  # original label
print(np.where(unique_labels == labels[0])[0][0])  # index where label occurs
print(boolean_labels[0].argmax())  # index where label occurs in boolean array
print(boolean_labels[0].astype(int))  # there will be a 1 where the sample label occurs

# Setup X & y variables
X = filenames
y = boolean_labels

print(f"Number of training images: {len(X)}")
print(f"Number of labels: {len(y)}")

NUM_IMAGES = 1000

# Split them into training and validation using NUM_IMAGES
X_train, X_val, y_train, y_val = train_test_split(X,
                                                  y,
                                                  test_size=0.1,
                                                  random_state=42)

print(
    len(X_train), len(y_train), len(X_val), len(y_val))

# Define image size
IMG_SIZE = 224


def Data_aug(image_path, label):
    """
        Takes an image file path and turns it into a Tensor.
        """
    # Read in image file
    image = tf.io.read_file(image_path)

    # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
    image = tf.image.decode_png(image, channels=3)

    # Convert the colour channel values from 0-225 values to 0-1 values
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

    # image = tf.image.central_crop(image, central_fraction=0.9)
    rand = tf.random.uniform([])
    rand1 = tf.random.uniform([])

    if rand > 0.5:
        # image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

    elif rand1 > 0.7:
        image = tf.image.random_flip_left_right(image)

        # image = tf.image.random_hue(image, 0.08)
        # image = tf.image.random_saturation(image, 0.6, 1.2)
        # image = tf.image.random_brightness(image, 0.1)
        # image = tf.image.random_contrast(image, 0.8, 1.2)
    else:
        image = image

    return image, label


def process_image(image_path):
    """
    Takes an image file path and turns it into a Tensor.
    """
    # Read in image file
    image = tf.io.read_file(image_path)

    # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
    image = tf.image.decode_png(image, channels=3)

    # Convert the colour channel values from 0-225 values to 0-1 values
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

    return image


# Create a simple function to return a tuple (image, label)
def get_image_label(image_path, label):
    """
    Takes an image file path name and the associated label,
    processes the image and returns a tuple of (image, label).
    """

    image = process_image(image_path)

    return image, label


def process_image1(image_path):
    """
    Takes an image file path and turns it into a Tensor.
    """
    # Read in image file
    image = tf.io.read_file(image_path)

    # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
    image = tf.image.decode_png(image, channels=3)

    # Convert the colour channel values from 0-225 values to 0-1 values
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

    return image


# Create a simple function to return a tuple (image, label)
def get_image_label1(image_path, label):
    """
    Takes an image file path name and the associated label,
    processes the image and returns a tuple of (image, label).
    """
    image = process_image1(image_path)

    return image, label


# Define the batch size, 32 is a good default
BATCH_SIZE = 32


def plot_images(dataset, n_images, samples_per_image):
    output = np.zeros((224 * n_images, 224 * samples_per_image, 3))

    row = 0
    for images, labels in dataset.repeat(samples_per_image).batch(n_images):
        output[:, row * 224:(row + 1) * 224] = np.vstack(images.numpy())
        row += 1

    plt.figure()
    plt.imshow(output)
    plt.show()


# Create a function to turn data into batches
def create_data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    """
    Creates batches of data out of image (x) and label (y) pairs.
    Shuffles the data if it's training data but doesn't shuffle it if it's validation data.
    Also accepts test data as input (no labels).
    """
    # If the data is a test dataset, we probably don't have labels
    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))  # only filepaths
        data_batch = data.map(process_image).batch(BATCH_SIZE)
        return data_batch

    # If the data if a valid dataset, we don't need to shuffle it
    elif valid_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x),  # filepaths
                                                   tf.constant(y)))  # labels

        data_batch = data.map(get_image_label1).batch(BATCH_SIZE)
        return data_batch

    else:
        # If the data is a training dataset, we shuffle it
        print("Creating training data batches...")
        # Turn filepaths and labels into Tensors

        data = tf.data.Dataset.from_tensor_slices((tf.constant(x),  # filepaths
                                                   tf.constant(y)))  # labels

        # Shuffling pathnames and labels before mapping image processor function is faster than shuffling images

        data = data.shuffle(buffer_size=len(x))

        # Make sure that the values are still in [0, 1]

        # Create (image, label) tuples (this also turns the image path into a preprocessed image)
        # Add augmentations

        

        data = data.map(
            lambda image, label: Data_aug(image, label)
            , num_parallel_calls=AUTOTUNE).repeat(2)

        # data = data.map(get_image_label, num_parallel_calls=AUTOTUNE)
        '''
        data_batch = data.map(
            lambda image, label: (process_image(image), label), num_parallel_calls=AUTOTUNE).cache().map(
            lambda image, label: Data_aug(image, label)
            , num_parallel_calls=AUTOTUNE).shuffle(len(x)).batch(
            BATCH_SIZE
        ).repeat(2).prefetch(AUTOTUNE)
        '''
        # data = data.map(Data_aug, num_parallel_calls=AUTOTUNE)

        # data = data.map(get_image_label)

        # data = data.map(get_image_label, num_parallel_calls=AUTOTUNE)
        # Add augmentations

        # Turn the data into batches
        data_batch = data.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return data_batch


# Create training and validation data batches
train_data = create_data_batches(X_train, y_train)
val_data = create_data_batches(X_val, y_val, valid_data=True)


# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied
# )

# Create a function for viewing images in a data batch
def show_25_images(images, labels):
    """
    Displays 25 images from a data batch.
    """
    # Setup the figure
    plt.figure(figsize=(20, 20))
    # Loop through 25 (for displaying 25 images)
    for i in range(25):
        # Create subplots (5 rows, 5 columns)
        ax = plt.subplot(5, 5, i + 1)
        # Display an image
        plt.imshow(images[i])
        # Add the image label as the title
        plt.title(unique_labels[labels[i].argmax()])
        # Turn gird lines off
        plt.axis("off")


# Visualize training images from the training data batch
train_images, train_labels = next(train_data.as_numpy_iterator())
# show_25_images(train_images, train_labels)
# plt.show()
# datagen.fit(train_images)

# Check out the different attributes of our data batches
print(train_data.element_spec, val_data.element_spec)

# Setup input shape to the model
INPUT_SHAPE = [None, 224, 224, 3]  # batch, height, width, colour channels

# Setup output shape of the model
OUTPUT_SHAPE = len(unique_labels)  # number of unique labels

# Setup model URL from TensorFlow Hub
MODEL_URL = "https://tfhub.dev/google/imagenet/resnet_v2_101/classification/4"


def get_pred_label(prediction_probabilities):
    """
    Turns an array of prediction probabilities into a label.
    """
    return unique_labels[np.argmax(prediction_probabilities)]


# Get a predicted label based on an array of prediction probabilities
# pred_label = get_pred_label(predictions[0])
# print(pred_label)

# Create a function to unbatch a batched dataset
def unbatchify(data):
    """
    Takes a batched dataset of (image, label) Tensors and returns separate arrays
    of images and labels.
    """
    images = []
    labels = []
    # Loop through unbatched data
    for image, label in data.unbatch().as_numpy_iterator():
        images.append(image)
        labels.append(unique_labels[np.argmax(label)])
    return images, labels


# Unbatchify the validation data
val_images, val_labels = unbatchify(val_data)
print(val_images[0], val_labels[0])
