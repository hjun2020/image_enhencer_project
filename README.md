# ESPCN implementation using darknet framework #
* Darknet is an open source neural network framework written in C and CUDA, https://github.com/pjreddie/darknet.
* The main purpose of Darknet is object detection, especially for a detection algorithm, YOLO.
* The goal of this project is to implement an image/video enhencing deep learning architecture using Darknet framework.
* Curretly, an example of ESPCN (https://arxiv.org/abs/1609.05158) algothm is implemented. 
* Source codes for most layers(CNN layer, cost layer, activation layer etc.) are from Darknet Framework.
* I added espcn layer(src/espcn_layer.c, src/espcn_layer.h) for image enhencement. 
* I also added some data preprocess/generation pipeline which are suitable for the image enhencing purpose.

## How to run network for some sample images
* You can try image enhencing for some sample images:
* Type ./darknet enhence cfg/enhencer2.cfg backup/enhencer2.backup_test3 data/eagle.jpg in the command line.
* The sample image could be data/dog.jpg, data/giraffe.jpg, data/kite.jpg etc.
* The original image will be downsized so that it has resolution 104 * 104. This is the input image of the network and saved at "input_image.jpg".
* The trained network will scale up the input image three times so the output image "output_image.jpg", has resolution 312 * 312.