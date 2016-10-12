#!/bin/sh

# Download Places365 model
wget http://places2.csail.mit.edu/models_places365/vgg16_places365.caffemodel
wget https://github.com/metalbubble/places365/raw/master/places365CNN_mean.binaryproto
wget https://raw.githubusercontent.com/metalbubble/places365/master/deploy_vgg16_places365.prototxt
wget https://raw.githubusercontent.com/metalbubble/places365/master/categories_places365.txt

# Download VGG16 model
wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
wget https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/0067c9b32f60362c74f4c445a080beed06b07eb3/VGG_ILSVRC_16_layers_deploy.prototxt
wget https://github.com/szagoruyko/imagenet-validation.torch/blob/master/data/VGG_mean.t7?raw=true
wget https://raw.githubusercontent.com/torch/tutorials/master/7_imagenet_classification/synset_words.txt
