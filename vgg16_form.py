from keras.applications.vgg16 import VGG16
from keras.layers import Dense, GlobalAveragePooling2D, merge
from keras.models import Sequential, Model
import glob
import numpy as np
from keras.optimizers import Adam
name_list = []
model = VGG16()


def vgg16(input_img):
    cou_f = 0
    cnn = VGG16(include_top=False, weights='imagenet', input_tensor=input_img)
    for dence in cnn.layers:
        for layer in cnn.layers:
            layer.trainable = False
        cou_f = cou_f + 1
        cnn.get_layer(name=dence.name).name = 'form' + str(cou_f)


    nn = cnn.output
    nn = GlobalAveragePooling2D(name="xcp_gap2d_form")(nn)
    nn = Dense(256, activation='relu', name='xcp_dense_relu_form')(nn)
    return nn

