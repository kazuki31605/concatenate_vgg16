import keras
from keras import backend as K, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import save_model, load_model
from keras.layers import BatchNormalization, Embedding, Concatenate, Maximum, Add
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils.np_utils import to_categorical
import os
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers import Activation, Dropout, Conv1D, Conv2D, Reshape, Lambda
import sys
from vgg16_bark import vgg16 as vgg16_bark
from vgg16_form import vgg16 as vgg16_leaf
from vgg16_leaf import vgg16 as vgg16_form



ROWS = 224
COLS = 224
CHANNELS = 3
label_list = []
image_list = []
# クラスの個数を指定
CLASSES = 3
batch = 32
epoch = 100

bark_label_list = []
leaf_label_list = []
form_label_list = []

bark_img_list = []
leaf_img_list = []
form_img_list = []

train_dir = 'C:/Users/hash/PycharmProjects/keras_deep/RGB-D/train/'

# path以下の*ディレクトリ以下の画像を読み込む。
for train_path in os.listdir(train_dir):
    for dir in os.listdir('{}{}'.format(train_dir, train_path)):
        # print('{}{}/{}'.format(train_dir, train_path, dir))
        for file in os.listdir('{}{}/{}'.format(train_dir, train_path, dir)):
            if str(train_path) == "depth":
                bark_label_list.append(int(dir) - 1)
                img = img_to_array(
                    load_img('{}{}/{}/{}'.format(train_dir, train_path, dir, file), target_size=(ROWS, COLS, CHANNELS)))
                bark_img_list.append(img)

            elif str(train_path) == "mas":
                form_label_list.append(int(dir) - 1)
                img = img_to_array(
                    load_img('{}{}/{}/{}'.format(train_dir, train_path, dir, file), target_size=(ROWS, COLS, CHANNELS)))
                form_img_list.append(img)

            elif str(train_path) == "rgb":
                leaf_label_list.append(int(dir) - 1)
                img = img_to_array(
                    load_img('{}{}/{}/{}'.format(train_dir, train_path, dir, file), target_size=(ROWS, COLS, CHANNELS)))
                leaf_img_list.append(img)

'''
bark_image_list = bark_image_list.flatten()
leaf_image_list = leaf_image_list.flatten()
form_image_list = form_image_list.flatten()
'''

Y = to_categorical(bark_label_list)


input_img_bark = Input(shape=(ROWS, COLS, CHANNELS), name="input_tensor_bark")
input_img_leaf = Input(shape=(ROWS, COLS, CHANNELS), name="input_tensor_leaf")
input_img_form = Input(shape=(ROWS, COLS, CHANNELS), name="input_tensor_form")
#input_img = Input(shape=(ROWS, COLS, CHANNELS), name="input_tensor_bark")
#


cnn1 = vgg16_bark(input_img_bark)
cnn2 = vgg16_leaf(input_img_leaf)
cnn3 = vgg16_form(input_img_form)

# leave only maximum features to eliminate null inputs
network = Concatenate()([cnn1, cnn2, cnn3])
network = Dense(4096, activation='relu', name="last_dense")(network)

# classification layer
network = Dropout(0.5)(network)
network = Dense(CLASSES, activation='softmax', name='softmax')(network)

model = Sequential()
model = Model(inputs=[input_img_bark, input_img_leaf, input_img_form], outputs=network)


model.compile(optimizer=Adam(lr=1e-4, decay=1e-6),
              loss=categorical_crossentropy,
              metrics=['accuracy'])

model.summary()  # モデルの表示

'''
bark_img_list = bark_img_list.flatten()
leaf_img_list = leaf_img_list.flatten()
form_img_list = form_img_list.flatten()
'''

# img_list = [bark_img_list, leaf_img_list, form_img_list]

# numpy配列に変更
bark_img_list = np.array(bark_img_list)
form_img_list = np.array(form_img_list)
leaf_img_list = np.array(leaf_img_list)

#img_list = [bark_img_list, leaf_img_list, form_img_list]



# 学習を実行。10%はテストに使用。
fit = model.fit(([bark_img_list, form_img_list, leaf_img_list]), Y,
                batch_size=batch, epochs=epoch, validation_split=0.1)

model.save('./my_model_RGB_D.h5')
model.save_weights('./my_model_weights_RGB_D.h5')

