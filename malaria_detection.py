%tensorflow_version 2.x
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
import cv2


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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
import keras.optimizers as optimizers
from keras.callbacks import ModelCheckpoint

from google.colab import files
files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!ls ~/.kaggle
!chmod 600 /root/.kaggle/kaggle.json
!kaggle datasets download -d iarunava/cell-images-for-detecting-malaria

!cd /content/
!unzip cell-images-for-detecting-malaria.zip

parasitized_data = os.listdir('../content/cell_images/cell_images/Parasitized')
print(parasitized_data[:10]) 

uninfected_data = os.listdir('../content/cell_images/cell_images/Uninfected')

print('\n')
print(uninfected_data[:10])

train_generator=ImageDataGenerator(rescale=1/255.0)
test_generator=ImageDataGenerator(rescale=1/255.0, validation_split=0.2)
datagen = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)

train_data=datagen.flow_from_directory(directory='/content/cell_images/cell_images', target_size=(180, 180), class_mode='binary', color_mode='rgb', batch_size=32,shuffle=True, subset='training')

test_data=datagen.flow_from_directory(directory='/content/cell_images/cell_images', target_size=(180, 180), class_mode='binary', color_mode='rgb', batch_size=32,shuffle=True, subset='validation')

model = Sequential()

#adding convolutional layers
model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=(180,180,3),activation='relu',padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(180,180,3),activation='relu',padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(filters=128, kernel_size=(3,3),input_shape=(180,180,3),activation='relu',padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(filters=256, kernel_size=(3,3),input_shape=(180,180,3),activation='relu',padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))

# flattening image
model.add(Flatten())

# adding dense layers
model.add(Dense(128,activation='relu'))
# adding dropout to minimize overfitting issue
model.add(Dropout(0.2))
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

#compiling the model
model.compile(loss='binary_crossentropy',optimizer='adam' ,metrics=["accuracy"])

#summary of the model
model.summary()

history = model.fit_generator(generator = train_data,steps_per_epoch = len(train_data),epochs =10, validation_data = test_data, validation_steps=len(test_data))
