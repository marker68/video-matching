from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, SpatialDropout3D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, \
    Convolution3D, MaxPooling3D, ZeroPadding3D
import cv2
import numpy as np


def vgg19(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3))) # Tensorflow order
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

def vgg16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def video_model_a(input_dim, output_dim):
    model = Sequential()
    model.add(SpatialDropout3D(0.2, input_shape=(5,input_dim,input_dim,3)))
    model.add(ZeroPadding3D((1, 1, 1)))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
    model.add(ZeroPadding3D((1, 1, 1)))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))

    model.add(ZeroPadding3D((1, 1, 1)))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
    model.add(ZeroPadding3D((1, 1, 1)))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))

    # not enough memory to run! :(
    # model.add(ZeroPadding3D((1, 1, 1)))
    # model.add(Convolution3D(256, 3, 3, 3, activation='relu'))
    # model.add(ZeroPadding3D((1, 1, 1)))
    # model.add(Convolution3D(256, 3, 3, 3, activation='relu'))
    # model.add(ZeroPadding3D((1, 1, 1)))
    # model.add(Convolution3D(256, 3, 3, 3, activation='relu'))
    # model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))
    #
    # model.add(ZeroPadding3D((1, 1, 1)))
    # model.add(Convolution3D(512, 3, 3, 3, activation='relu'))
    # model.add(ZeroPadding3D((1, 1, 1)))
    # model.add(Convolution3D(512, 3, 3, 3, activation='relu'))
    # model.add(ZeroPadding3D((1, 1, 1)))
    # model.add(Convolution3D(512, 3, 3, 3, activation='relu'))
    # model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=output_dim, activation='linear'))

    return model


def video_model_b(input_dim, output_dim):
    model = Sequential()
    model.add(SpatialDropout3D(0.2, input_shape=(5,input_dim,input_dim,3)))
    model.add(ZeroPadding3D((1, 1, 1)))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
    model.add(ZeroPadding3D((1, 1, 1)))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))

    model.add(SpatialDropout3D(0.2))
    model.add(ZeroPadding3D((1, 1, 1)))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
    model.add(ZeroPadding3D((1, 1, 1)))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=output_dim, activation='linear'))
    return model


def VideoTextModel(input_dim, output_dim, name='a'):
    if name == 'a': model = video_model_a(input_dim=input_dim, output_dim=output_dim)
    elif name == 'b': model = video_model_b(input_dim=input_dim, output_dim=output_dim)

    return model


def load_model(model_file, type='vgg19'):
    if type == 'vgg19':
        model = vgg19(model_file)
    elif type == 'vgg16':
        model = vgg16(model_file)
    return model


def load_images(imgs):
    ims = []
    for img in imgs:
        im = cv2.resize(cv2.imread(img), (224, 224)).astype(np.float32)
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, axis=0)
        ims.append(im)
    return ims


def savelabels(labels, output):
    f = open(output, 'w')
    for label in labels:
        f.write(label + "\n")
    f.close()


def classify(model, ims, synsets):
    top_ks = []
    for im in ims:
        out = model.predict(im)
        top_k = out.flatten().argsort()[-1:-6:-1] # top 5
        for k in top_k: top_ks.append(k)

    # print predicted labels
    labels = np.loadtxt(synsets, str, delimiter='\t')

    lbl = []
    for k in top_ks:
        lbl.append(labels[k])
    return lbl
