#!/usr/bin/env python3
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Convolution2D, Flatten, Dropout, Dense
from keras.callbacks import ModelCheckpoint

'''
This scripts retrains and saves a VGG16 neural network pretrained
on ImageNet and tries to predict the type of cervix from an image
'''

print("Loading dataset...")
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

generator = datagen.flow_from_directory(
    'datasets/train512',
    target_size=(224, 224),
    batch_size=16,
    shuffle=False)

print("Loading VGG16 model...")
body = VGG16(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
head = body.output
head = Flatten()(head)
head = Dropout(0.3)(head)
head = Dense(512, activation='relu')(head)
head = Dropout(0.3)(head)
head = Dense(256, activation='relu')(head)
head = Dense(3, activation='softmax')(head)

model = Model(body.input, head)

# Freezing layers
for layer in model.layers[:19]:
    print("Disabled: ", layer)
    layer.trainable = False

model.summary()

print("Training...")
model.compile(optimizer='adagrad',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

filepath = "pretrained/cervix_vgg16-{epoch:02d}-{loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

model.fit_generator(generator, samples_per_epoch=1024, nb_epoch=50, verbose=2, callbacks=callbacks_list)

model.save("pretrained/cervix_vgg16.h5")
