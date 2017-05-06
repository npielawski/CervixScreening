#!/usr/bin/env python3
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, AveragePooling2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.callbacks import ModelCheckpoint

'''
This scripts creates, trains and saves a simple convolutional neural
network and tries to predict the type of cervix from an image
'''

# Parameters
dataset_train = 'datasets/train512'
dataset_val = 'datasets/validation512'
batch_size = 64
print("Loading dataset...")
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.4,
    zoom_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

generator = datagen.flow_from_directory(
    dataset_train,
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=True)

val_datagen = ImageDataGenerator()
val_generator = val_datagen.flow_from_directory(
    dataset_val,
    target_size=(224, 224),
    batch_size=batch_size)

# '''
print("Creating model...")
model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(224, 224, 3)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
model.add(AveragePooling2D(pool_size=(6, 6)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

print("Layers count:", len(model.layers))

model.summary()

print("Training...")
model.compile(optimizer='adagrad',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

filepath = "pretrained/cervix_cnn-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

model.fit_generator(
    generator,
    samples_per_epoch=1024,
    nb_epoch=50,
    validation_data=val_generator,
    nb_val_samples=148 // batch_size,
    verbose=2,
    callbacks=callbacks_list
)

model.save("pretrained/cervix_cnn.h5")
