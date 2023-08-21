## Notebooks quick presentation

Notebooks are organized within 3 folders following the development logic of the project:

#### 1. Exploratory Data Analysis:

    - Creation of a Pandas dataframe gathering useful data about the plant pictures, including filepaths and labels
    - Dataset analysis using data visualization tools
    
#### 2. Modelling:
This folder gathers the notebooks used to develop our Computer Vision models.  
This phase of the project was divided into three main steps:
    - First models: training and test of several first CNN models: one basic model based on Lenet-5 architecture, and three Transfer Learning models based on different architectures (VGG19, MobileNetV2 and ResNet50) pre-trained on ImageNet database. This folder also includes a notebook dedicated to model interpretation using Grad-CAM tool (for VGG19 model only).
    -  Optimization: two main avenues were investigated in order to improve the models:
        - Limitation of data augmentation parameters: first models were trained with strong image augmentation parameters in the pre-processing step (shear, rotation, zoom and translation shift). We assumed that the important modifications produced on images by such a strong data augmentation may limit the training quality of the models, and thus decided to re-train the models with more limited image augmentation.
        - Image segmentation: the idea here was to train the models on segmented images, in order to both prevent any data leakage from the pictures' background, and improve models accuracy trough an easier plant contours identification.  
        We first used colorimetric thresholding to segment the images: a colorimetric analysis of the dataset allowed us to select a convenient threshold and colorspace, and then we re-trained the models including a thresholding-based segmentation step in the pre-processing phase.  
        This thresholding-based segmentation approach was however limited, because not applicable to images with background and plant of similar colours. This led us to train a deep learning segmentation model based on Unet architecture, so as to free ourselves from the colorimetric constraints inherent to thresholding.
    - Models analysis: in this part we assess and compare the different models' accuracy, and evaluate the respective impact of the optimization attempts.

#### 3. Predictions:
In this notebook we use the three Transfer Learning optimized models (VGG19, MobileNetV2 and ResNet50) to predict plant seedlings specie from single images. This code was used as a basis to present the results of our models in the Streamlit app. **It has to be noted that our ResNet50 model was too heavy to be uploaded on Github, so unfortunately  the parts of the notebook using this model cannot be run.**
    - The first part of the notebbok is dedicated to checking the consistency between the predictions obtained with two different pre-processing approaches: manual pre-processing on the one hand, and pre-processing using Keras ImageDataGenerator class on the other hand (this point had indeed posed some difficulties).
    - In the following section we compare the predictions obtained by MobileNet model with both segmentation techniques: thresholding-based segmentation and Deep Learning (Unet) segmentation.
    - The last part of the notebook adresses the comparison of the three models' predictions on single images: first on images randomly drawn from the dataset, then on external images found on the web.
        

