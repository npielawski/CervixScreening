#!/usr/bin/env python3
import numpy as np
from skimage.measure import label
from skimage.morphology import erosion, square


# Generates the heat map thanks to the neural network skin_detector
# patchw and patchh are resized for the neural network input
def generate_skin_heatmap(model, image, patchw=64, patchh=64):
    imw, imh = image.size
    # Find the width and height of the heat map and allocate the memory
    heatw, heath = int(imw / patchw), int(imh / patchh)
    # We create the batch
    print("Generating batch...")
    batch = []
    for x in range(heatw):
        for y in range(heath):
            # We crop the image
            crop = image.crop((x*patchw, y*patchh, (x+1)*patchw, (y+1)*patchh))
            # We adapt the patch to the input size of the neural network
            downsampled = crop.resize(model.input_shape[1:3])
            # We adapt the axes to tensorflow
            batch.append(np.array(downsampled))
    batch = np.array(batch)
    print("Batch shape:", batch.shape)
    # Then we process it all
    print("Generating heatmap...")
    heatmap = np.array(model.predict(batch))
    heatmap.resize([heatw, heath])
    heatmap = heatmap.swapaxes(0, 1)
    print("Heatmap shape:", heatmap.shape)
    return heatmap


# Extracts the patch from the picture, thanks to the
# heatmap. It takes the biggest blob and surround it
# with a rectangle.
def extract_image_from_heatmap(image, heatmap, thresh=0.5):
    print("Finding extraction area...")
    # Let's find the best area to extract
    thresholded = heatmap > thresh
    filtered = erosion(thresholded, square(3))
    # if nothing triggered the filter
    if filtered.sum() == 0:
        filtered.fill(1.)
    # Let's save the biggest blob only and discard the rest
    labeled, labelcount = label(filtered, return_num=True, connectivity=1)
    print("Number of blobs found:", labelcount)
    # For each blob, we try to figure out the biggest
    biggestblobsize = 0
    biggestblobid = -1
    for i in range(1, labelcount+1):
        blobsize = (thresholded * (labeled == i)).sum()
        if blobsize >= biggestblobsize:
            biggestblobid = i
            biggestblobsize = blobsize
    print("Biggest blob size: {} (id: {})".format(biggestblobsize, biggestblobid))
    # Remaining blob
    biggestblob = (labeled == biggestblobid)
    # Finding the position of the rectangle
    xs, ys = np.indices(biggestblob.shape)
    xi, yi = xs[biggestblob != 0], ys[biggestblob != 0]
    rectx1, rectx2 = xi.min(), xi.max() + 1
    recty1, recty2 = yi.min(), yi.max() + 1
    # Rescale to the big image
    ratiox = image.size[0] / thresholded.shape[1]
    ratioy = image.size[1] / thresholded.shape[0]
    srectx1, srectx2 = int(rectx1 * ratiox), int(rectx2 * ratiox)
    srecty1, srecty2 = int(recty1 * ratioy), int(recty2 * ratioy)
    # Extraction of the rectangle
    print("Heatmap rectangle: ", [rectx1, rectx2, recty1, recty2])
    print("Scaled rectangle: ", [srectx1, srectx2, srecty1, srecty2])
    extracted = np.array(image)[srectx1:srectx2, srecty1:srecty2]
    return extracted, labeled, thresholded
