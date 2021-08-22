# -*- coding: utf-8 -*-
"""Malaria_Detection.ipynb
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
!pip install -q kaggle
from urllib.request import urlretrieve
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import pandas as pd
import tensorflow as tf
import seaborn as sns
import os
import random
from sklearn.model_selection import train_test_split



import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds



from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape, Dense, Conv2D, GlobalAveragePooling2D, BatchNormalization
from keras.regularizers import l2
from keras.wrappers.scikit_learn import KerasClassifier
import keras.optimizers as optimizers
from keras.callbacks import ModelCheckpoint

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.densenet import DenseNet121
!pip install kaggle

from google.colab import files
files.upload()
!ls -lha kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d iarunava/cell-images-for-detecting-malaria

!cd /content/
!unzip cell-images-for-detecting-malaria.zip

imageDataGenerator = tf.keras.utils.image_dataset_from_directory
batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "/content/cell_images", 
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
train_ds=train_ds.shuffle(300)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "/content/cell_images",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
val_ds=val_ds.shuffle(300)

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(32,(5, 5), padding='same', activation='relu'),
  layers.Conv2D(32,(5, 5), padding = 'same', activation ='relu'),
  layers.MaxPool2D(),
  layers.Dropout(0.25),

  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPool2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPool2D(),
  layers.Dropout(0.25),

  layers.Flatten(),
  layers.Dense(256, activation='softmax'),
  layers.Dropout(0.5),
  layers.Dense(10),
])

opt = keras.optimizers.SGD(learning_rate=0.01, decay=1e-6)

    # train the model using SDG(Stochastic Gradient Descent)
model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=[tf.keras.metrics.AUC()])

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model.summary()

epochs=1
history = model.fit(train_ds,validation_data=val_ds, epochs=epochs, verbose=2)
