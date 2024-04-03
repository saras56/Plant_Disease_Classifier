
# **Multiclass Plant Disease Classification using CNN**

This repository contains the code for a multiclass classification model trained to classify potato plant leaf diseases into 3 categories : Late Blight, Early Blight and Healthy(no disease). The model architecture used for this classification task is classic convolutional neural network 

## **Problem Statement**

Early detection and accurate diagnosis of leaf diseases are crucial for effective disease management and prevention of its spread. The traditional methods of disease diagnosis rely on visual inspection by experts which is time-consuming and can be prone to errors. In case of potato plants, the diseases earlt blight and late blight are the most frequent. As the treatment for these diseases are different it is important to identify them accurately and early so as to minimize the economical loss by farmers. Here, a simple CNN model is implemented to classify the diseased plants.

## **Dataset**
The dataset used for this model is taken from Plant Village Dataset available on Kaggle. There were a total of 2152 images belonging to three classes namely Late Blight ,Early Blight ,Healthy

<p align="center">
  <img src="https://github.com/saras56/Plant_Disease_Classifier/assets/115695360/61799960-a02f-4f8f-a865-9712180b00ce" alt="Description of the image">
</p>
<p align="center">
  Sample Images from dataset
</p>

## **Splitting the data into train, test and validation**
Here the train data is split into train and validation sets using split-folders package from python. The split is done in such a way that 70% of the data goes to training, 10% for validation and remaining 20% for test.  The test data is completely unseen. There are 1506 train images, 215 validation images an 431 test images.   

## **Image Augmentation using Image Data Generator**
Data Augmentation is a process that generates several realistic variants of each training sample, to artificially expand the size of the training dataset. This aids in the reduction of overfitting. In data augmentation, we will slightly shift, rotate, and resize each image in the training set by different percentages, and then add all of the resulting photos to the training set. This allows the model to be more forgiving of changes in the object’s orientation, position, and size in the image.
The augmentation that have been used on the training dataset is not applied on validation and testing data as the validation and testing dataset will only test the performance of the model, and based on it, the  model parameters or weights will get tunned. Our objective is to create a generalized and robust model, which we can achieve by training our model on a very large amount of dataset. That’s why here we are only applying data augmentation on the training dataset and artificially increasing the size of the training dataset. The taget size of the images are 256*256. 

## **Model Training and Results**
A simple and classical Convolutional Neural Network model is used to train the dataset. The following results have been achieved with the CNN model for detection Potato plant disease

- Test Accuracy		  : 95%
- f1-score (Early_blight)	  : 96%
- f1-score (Late_blight)	  : 94%
- f1-score (Healthy)	  : 94%

**Confusion matrix**

<p align="center">
  <img src="https://github.com/saras56/Plant_Disease_Classifier/assets/115695360/50cfcd47-f761-4065-b12b-cf33ec5ebe6f">
</p>

**Sample predictions**

<p align="center">
  <img src="https://github.com/saras56/Plant_Disease_Classifier/assets/115695360/676ff1cd-5637-4bb0-990a-48c236783b8f">
</p>

## **Inference**
The API for the model is build and tested using Postman. 
For example: For the given Early_Blight image, the prediction on postman is given below

<p align="center">
  <img src="https://github.com/saras56/Plant_Disease_Classifier/assets/115695360/2a95a909-d3e9-4adb-8423-82495b6ed118" alt="Description of the image">
</p>
<p align="center">
  Potato leaf with Early Blight disease
</p>

<p align="center">
  <img src="https://github.com/saras56/Plant_Disease_Classifier/assets/115695360/613318ac-15b4-4f33-a439-b82f577575c2" alt="Description of the image">
</p>
<p align="center">
  Postman interface
</p>

## **Streamlit App**

<p align="center">
  <img src="https://github.com/saras56/Plant_Disease_Classifier/assets/115695360/aba55de5-a648-4477-9124-7d22463157c8" alt="Description of the image">
</p>
<p align="center">
  Prediction for Late Blight Disease
</p>

## **Future work**
- Increase the number of samples in the dataset
- Try out a different architecture for CNN to improve the results
