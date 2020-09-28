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

# create model
from tensorflow.python.keras.layers import MaxPooling2D

from CreateModelCovid import *

''' FINE_Tuning
     stin ousia apenergopoiw ola ta layers apla afhnw ta 2 teleytai aki ta ekpadeyv
    baseModel = tf.keras.applications.resnet50.ResNet50(weights="imagenet", include_top=False,
                                                     input_tensor=Input(shape=(224, 224, 3)))
       headModel = baseModel.output
       headModel = Flatten(name="flatten")(headModel)
       headModel = Dense(64, activation="relu")(headModel)
       headModel = Dropout(0.5)(headModel)
       headModel = Dense(3, activation="softmax")(headModel)
       model = Model(inputs=baseModel.input, outputs=headModel)
       for layer in baseModel.layers:
           layer.trainable = False
       model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

       apo katw einai transfer learning 
    model.trainable = False
    inputs = keras.Input(shape=(224, 224, 3))
    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`. This is important for fine-tuning, as you will
    # learn in a few paragraphs.
    x = model(inputs, training=False)
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = keras.layers.GlobalAveragePooling2D()(x)
    # A Dense classifier with a single unit (binary classification)
    outputs = keras.layers.Dense(3)(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.add(Dense(3, activation='softmax'))
    model.layers[0].trainable = False
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    '''

data_augmentation = Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal",
                                                 input_shape=(224,
                                                              224,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),

  ]
)
# Create a function which builds a Keras model
def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
    print("Building model with:", MODEL_URL)

    '''
    # Xception
    model.add(tf.keras.applications.Xception(include_top=False, pooling='avg', weights='imagenet',
                                             input_shape=(224, 224, 3))
    '''
    '''
    model.add(tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet',
                                                             input_shape=(224, 224, 3), classes=3))
    '''

    for image_batch, label_batch in train_data.take(-1):
        pass

    # model = tf.keras.applications.MobileNet(include_top=False, weights='imagenet',
    # input_shape=(224, 224, 3))
    '''
    fbatch = model(image_batch)
    model.trainable = False
    model.summary()
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(fbatch)

    prediction_layer = tf.keras.layers.Dense(3)
    prediction_batch = prediction_layer(feature_batch_average)
    model = tf.keras.Sequential([
        model,
        global_average_layer,
        prediction_layer
    ])
    '''
    #model = Sequential([data_augmentation])

    model = Sequential()
    model.add(tf.keras.applications.MobileNetV2(include_top=False,
                                                weights='imagenet',
                                                input_shape=(224, 224, 3)))
    for layer in model.layers:
        layer.trainable = False
    # top layer for shaping output
    headModel = model.output
    headModel = MaxPooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(64, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)

    headModel = Dense(3, activation="softmax")(headModel)
    model = Model(inputs=model.input, outputs=headModel)

    # model.layers[0].trainable = False

    base_learning_rate = 0.0010
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate, decay=base_learning_rate / 20),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model


# Create a model and check its details
model = create_model()
model.summary()

# Create early stopping (once our model stops improving, stop training)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                  patience=3)  # stops after 3 rounds of no improvements

# How many rounds should we get the model to look through the data?
NUM_EPOCHS = 20
# se ola ta data train
full_data = create_data_batches(X, y)

full_model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor="accuracy",
                                                             patience=10)
'''# Create training and validation data batches
train_data = create_data_batches(X_train, y_train)
val_data = create_data_batches(X_val, y_val, valid_data=True)'''


# Build a function to train and return a trained model
def train_model():
    """
    Trains a given model and returns the trained version.
    """
    # Create a model
    model = create_model()

    # Fit the model to the data passing it the callbacks we created
    model.fit(create_data_batches(X_train, y_train),
              epochs=NUM_EPOCHS,
              validation_data=val_data,
              validation_freq=1)
              #,callbacks=[])

    return model


# Fit the model to the data
model = train_model()
history2 = model

# Plotting train_loss vs val_loss
plt.figure(figsize=(10, 5))
plt.plot(model.history.history["loss"], label="loss")
# plt.plot(model.history .history["val_loss"], label="val_loss")
plt.legend()

# Plotting train_accuracy vs Val_accuracy
plt.figure(figsize=(10, 5))
plt.plot(model.history.history["accuracy"], label="accuracy")
# plt.plot(model.history.history["val_accuracy"], label="val_accuracy")
plt.legend(loc='upper left')

plt.show()


def save_model(model, suffix=None):
    """
    Saves a given model in a models directory and appends a suffix (str)
    for clarity and reuse.
    """
    # Create model directory with current time
    modeldir = os.path.join("C:/Users//giorgos//Desktop//576013_1042828_compressed_COVID-19 Radiography Database (2)//")
    model_path = modeldir + "-" + suffix + ".h5"  # save format of model
    print(f"Saving model to: {model_path}...")
    # tf.saved_model.save(model, model_path)
    model.save(model_path)
    # tf.compat.v1.keras.experimental.export_saved_model(model, model_path)
    return model_path


# Save our model trained on 1000 images
saved_full_image_model_path = save_model(model, suffix="full-covid-model_teliko")
