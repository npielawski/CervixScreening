#!/usr/bin/env python3
import os
from PIL import Image
from keras.models import load_model
import numpy as np

'''
Generates the final resulting file to send to Kaggle.
Last step.
'''

infolder = "datasets/test512"

# Loading of the skin detection neural network
cervixmodelpath = "pretrained/cervix_best.h5"
cervixmodel = load_model(cervixmodelpath)

f = open("kaggle-submission.csv", "w+")
f.write("image_name,Type_1,Type_2,Type_3\r\n")

# Beginning of the treatment of the dataset
# I would advice you to take many coffees
# and go on reddit.com/r/machinelearning
# It takes a while.
for subdir, dirs, files in os.walk(infolder):
    for file in files:
        imagepathin = os.path.join(subdir, file)

        # We can start
        print("Loading image:", imagepathin)
        image = Image.open(imagepathin)
        try:
            image.load()
        except OSError:
            print("Corrupted image:", imagepathin)
            continue
        image = image.resize((224, 224))

        prediction = cervixmodel.predict(np.asarray(image).reshape((1, 224, 224, 3))).flatten()
        print("prediction:", prediction)

        filename = os.path.basename(imagepathin)
        # Removing leading 0's
        filename = str(int(filename.split('.')[0])) + ".jpg"
        output = filename + "," + ",".join(map(str, prediction))
        print("kaggle output:", output)
        f.write(output + "\r\n")

        print()
print("Finished! Good luck :)")