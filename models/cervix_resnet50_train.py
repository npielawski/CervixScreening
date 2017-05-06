#!/usr/bin/env python3
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Flatten, Dropout, Dense
from keras.callbacks import ModelCheckpoint

'''
This scripts retrains and saves a residual network pretrained
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
    'datasets/additional512',
    target_size=(224, 224),
    batch_size=64,
    shuffle=True)

val_datagen = ImageDataGenerator()
val_generator = val_datagen.flow_from_directory(
    'datasets/train512',
    target_size=(224, 224),
    batch_size=64)

# '''
print("Loading VGG16 model...")
body = ResNet50(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
head = body.output
head = Flatten()(head)
head = Dropout(0.3)(head)
head = Dense(3, activation='softmax')(head)

model = Model(body.input, head)

print("Layers count:", len(model.layers))
# Freezing layers
for layer in model.layers[:177-1]:
    print("Disabled: ", layer)
    layer.trainable = False

model.summary()

print("Training...")
model.compile(optimizer='adagrad',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
'''
from keras.models import load_model

print("Loading model...")
model = load_model("pretrained/cervix_best2.h5")

for layer in model.layers:
    layer.trainable = True

model.summary()

print("Training...")
model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9, decay=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

'''

filepath = "pretrained/cervix_resnet50-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

model.fit_generator(
    generator,
    samples_per_epoch=1024,
    nb_epoch=50,
    validation_data=val_generator,
    nb_val_samples=148 // 64,
    verbose=2,
    callbacks=callbacks_list
)

model.save("pretrained/cervix_vgg16.h5")
