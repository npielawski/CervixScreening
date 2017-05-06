#!/usr/bin/env python3
import os
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D

'''
This script trains a simple convolutional neural network from the skindataset.
The purpose is to generate a heatmap of the the cervix area.
More info in notebooks/Cervix Extract.pdf
'''

# Extracts a patch from the image at a random position
# The size of the patch is patch_width*patch_height (default: 16x16)
# If there is too much white in the picture, the patch is dropped and
# another one is chosen. Percentage between 0. and 1. (default: 90%)
def get_random_crop_from_image(image, patch_width=16, patch_height=16, drop_white=0.9):
    width, height = image.size
    x = np.random.randint(width - patch_width)
    y = np.random.randint(height - patch_height)
    crop = np.array(image.crop((x, y, x + patch_width, y + patch_height)))
    # We drop pictures that have too much white in them
    white_qty = np.count_nonzero(crop.mean(axis=2) == 255) / (patch_width * patch_height)
    if white_qty >= drop_white:
        return get_random_crop_from_image(image, patch_width, patch_height, drop_white)
    return crop


# Generate a new training sample
def generate_training_sample(patch_width=16, patch_height=16):
    # We chose a random category (either nonskinim or skimim)
    label = np.random.randint(2)
    category = nonskinim if label == 0 else skinim
    # We take a random picture from the chosen category
    picture = category[np.random.randint(len(category))]
    crop = get_random_crop_from_image(picture, patch_width, patch_height)
    return crop, label


# Generator for keras
def sample_generator(batch_size=32, patch_width=16, patch_height=16):
    while 1:
        x = []
        y = []
        for i in range(batch_size):
            crop, label = generate_training_sample(patch_width, patch_height)
            x.append(crop)
            y.append(label)
        yield (np.array(x), np.array(y))

if __name__ == "__main__":
    # List non-skin images
    nonskinfile = []
    nonskinim = []
    for subdir, dirs, files in os.walk("datasets/skindataset/nonskin"):
        for file in files:
            path = os.path.join(subdir, file)
            nonskinfile.append(path)
            nonskinim.append(Image.open(path))
    print("Non-skin samples:", len(nonskinfile))
    # List skin images
    skinfile = []
    skinim = []
    for subdir, dirs, files in os.walk("datasets/skindataset/skin"):
        for file in files:
            path = os.path.join(subdir, file)
            skinfile.append(path)
            skinim.append(Image.open(path))
    print("Skin samples:", len(skinfile))

    # Creation of the model
    print("Building model...")
    model = Sequential()

    model.add(Convolution2D(8, 3, 3, activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(16, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(16, 3, 3, activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    print()
    print("Starting the training...")

    model.compile(optimizer='adadelta',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit_generator(sample_generator(32, 64, 64), samples_per_epoch=1024, nb_epoch=30, verbose=2)

    model.save("pretrained/skincolor.h5")
