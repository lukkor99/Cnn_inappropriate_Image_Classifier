# Cnn inappropriate Image Classifier | Pytorch


## Project description
<p align="justify">This is multiclass classifier written in Python3. I've created a Convolutional Neural Network to classify images to one of three target class:
<p>
<ol>
<li> porn/nudity images
<li> violence images
<li> standard images
</ol>
<p align="justify"> Basic accuracy was about 60%. I've reached evaulation accuracy level around 93%. Model was overfitted and biased for only porn image class. 
I've used several techniques to avoid overfitting, minimaze loss and improve accuracy of each image class: <p>

* expanding dataset size
* data augmentation - additional transforms
* learning rate scheduling
* testing various model's architectures
* dropouts

## Flask App
<p align="justify">
I've wrote simple app which can use model's file with .pth extension. Predicting method requests for image file with three most common image extensions: .png, .jpeg and .jpg.
File validation is included, predicting function returns predicted class and class name.

## Motivation
This project was created certainly in education purposes. All the research about datasets and deep learning methods didn't bring me any benefitts.

## Libraries

* Pytorch
* Torchvision
* OS
* Flask
* PIL
* IO

## References for dataset
 
* [PornHub images meta-data dataset](https://github.com/cdipaolo/hub-db)
* [Google Open Images](https://opensource.google/projects/open-images-dataset)
* [M. Bianculli, N. Falcionelli, P. Sernani, S. Tomassini, P. Contardo, M. Lombardi, A.F. Dragoni, A dataset for automatic violence detection in videos, Data in Brief 33 (2020). doi:10.1016/j.dib.2020.106587 - frames has been extracted from videos](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos)
* [Unsplash Stock Images](https://unsplash.com/s/photos/violence)

## References for research

* [Build an Image Classification Model using Convolutional Neural Networks in PyTorch](https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/)
* [TRAINING A CLASSIFIER - Pytorch Tutorials](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
* [Pytorch Documentation](https://pytorch.org/docs/stable/index.html)
* [Getting Started with Albumentation: Winning Deep Learning Image Augmentation Technique in PyTorch example](https://towardsdatascience.com/getting-started-with-albumentation-winning-deep-learning-image-augmentation-technique-in-pytorch-47aaba0ee3f8)
* [Nudity Detection and Abusive Content Classifiers — Research and Use cases](https://towardsdatascience.com/nudity-detection-and-abusive-content-classifiers-research-and-use-cases-d8886bf624e8)

##
