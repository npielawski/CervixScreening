# Kaggle competition: Cervical Cancer Screening
_Competition deadline: June 21, 2017_

## Overview
### Data
In this competition, you will develop algorithms to correctly classify cervix types based on cervical images. These different types of cervix in our data set are all considered normal (not cancerous), but since the transformation zones aren't always visible, some of the patients require further testing while some don't. This decision is very important for the healthcare provider and critical for the patient. Identifying the transformation zones is not an easy task for the healthcare providers, therefore, an algorithm-aided decision will significantly improve the quality and efficiency of cervical cancer screening for these patients.

### Additional information
<https://kaggle2.blob.core.windows.net/competitions/kaggle/6243/media/Cervix%20types%20clasification.pdf>

## Project structure
Different folders are available:
* datasets: contains the different datasets
* models: contains the trained neural networks.
* notebooks: contains explanations and trials about this contest
* pretrained: contains the neural networks that have been trained
* scripts: contains different tools to be used by models and notebooks

### Running the files
In the main folder, use the command 'python3 -m'

Example for extraction the cervix region from a dataset and resizing to 512x512:
* python3 -m scripts.skincolor_createdataset datasets/additional datasets/additional512 512
Example for training neural networks:
* python3 -m models.cervix_cnn_train
* python3 -m models.cervix_resnet50_train

### File standard
* For models:
    * \<project\>\_\<networktype\>\_train to train the model on the specified dataset.
    * \<project\>\_\<networktype\>\_test to test the model on the specified dataset.
    * \<project\>\_\<networktype\>\_predict to predict a single outcome.

* For scripts:
    * \<project\>_preparation is data preparation for the exclusive use of a neural network.

## Ideas

Many pictures contain two areas: the cervix part and the part around. The
second part can be skin, clothes, tools and is irrelevant for the classification.
The idea developed is the creation of a convolutional neural network that will
generate a heat map of the cervix-like parts of the images. An example is shown
in the notebook: Cervix Extraction.pdf. Once extracted, I retrained a resnet50 over
the cropped cervix and resulting in a Kaggle loss of 0.87.

## Scores
| ID | Neural network        | Dataset            | Test set loss | Kaggle loss |
|----|-----------------------|--------------------|---------------|-------------|
| 1  | Resnet50 (last layer) | cervix_cnn 512x512 | 0.43          | 0.8789      |

1) cervix_cnn dataset 512px, Resnet50 with last layer trained with Adagrad
