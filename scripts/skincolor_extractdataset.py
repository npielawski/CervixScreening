#!/usr/bin/env python3
import os
import sys
from PIL import Image
from keras.models import load_model
from models import skincolor as skin

'''
This scripts generates a cropped image around the cervix area from another
image file. It can also resize the resulting image (optional)
The scripts applies the same treatment to all the files available in the
given folder.
'''

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Missing argument:")
        print("python3 createfocusdataset.py <in folder> <out folder> <size of image>")
        print("The size of the output image is optional. If not set, the patches will")
        print("keep their original size")
        sys.exit(0)

    if len(sys.argv) > 3:
        imsize = int(sys.argv[3])
    else:
        imsize = -1

    infolder = sys.argv[1]
    outfolder = sys.argv[2]

    # Creation of the folder if it does not exist
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    else:
        res = input("Output folder already exist, is it normal? (Y/N): ")
        if res.upper() != 'Y':
            print("Exiting...")
            sys.exit(0)

    # Loading of the skin detection neural network
    skinmodelpath = "pretrained/skincolor.h5"
    skinmodel = load_model(skinmodelpath)

    # Beginning of the treatment of the dataset
    # I would advice you to take many coffees
    # and go on reddit.com/r/machinelearning
    # It takes a while.
    for subdir, dirs, files in os.walk(infolder):
        for file in files:
            imagepathin = os.path.join(subdir, file)
            # Removal of the infolder path part, then split
            pathsplit = imagepathin[len(infolder)+1:].split("/")
            # Replacement of the folder with the export one
            pathsplit.insert(0, outfolder)
            # Padding of the output file
            filenamesplit = pathsplit[-1].split(".")
            # We drop weird files (starting with .)
            if not filenamesplit[0]:
                continue
            newfilename = "%04d.%s" % (int(filenamesplit[0]), filenamesplit[1])
            pathsplit[-1] = newfilename
            imagepathout = '/'.join(pathsplit)
            # We create the associated folder if necessary
            if not os.path.exists(os.path.dirname(imagepathout)):
                os.makedirs(os.path.dirname(imagepathout))
            # At this point we have two relevant variables:
            # - imagepathin: the image to transform and rescale
            # - imagepathout: the output image to create

            # If the file exists, let's go to the next one
            if os.path.exists(imagepathout):
                print("Existing image:", imagepathout)
                continue

            # We can start
            print("Loading image:", imagepathin)
            try:
                image = Image.open(imagepathin)
            except:
                print("Cannot open image:", imagepathin)
            try:
                image.load()
            except OSError:
                print("Corrupted image:", imagepathin)
                continue

            heatmap = skin.generate_skin_heatmap(skinmodel, image)
            extracted, filtered, thresholded = skin.extract_image_from_heatmap(image, heatmap)

            # Neural network image input
            neuralimage = Image.fromarray(extracted)
            if imsize > 0:
                neuralimage = neuralimage.resize((imsize, imsize), Image.BICUBIC)

            neuralimage.save(imagepathout)
            print("Output image size:", neuralimage.size)
            print("Saving image:", imagepathout)
            print()
    print("Finished! Have fun :)")