# ESPCN implementation using darknet framework #
* Darknet is an open source neural network framework written in C and CUDA, https://github.com/pjreddie/darknet
* The main purpose of Darknet is object detection, especially for a detection algorithm, YOLO
* The goal of this project is to implement an image/video enhencing deep learning architecture using Darknet framework.
* Curretly, an example of ESPCN (https://arxiv.org/abs/1609.05158) algothm is implemented. 
* Source codes for most layers(CNN layer, cost layer, activation layer etc.) are from Darknet Framework
* I added espcn layer(src/espcn_layer.c, src/espcn_layer.h) for image enhencement. 
* I also added some data generation code which are suitable for the image enhencing purpose

