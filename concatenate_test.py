#Cording: UTF-8

from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.initializers import TruncatedNormal, Constant
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
from PIL import Image
import os
import sys
import glob

#分類先ディレクトリ構造
#親ディレクトリの下にクラスの数だけ作り，クラス番号を1から順に割り振る
#その下に分類対象の画像を格納

label_list = []
#5クラスなので5*5で25を確保している
total = 0.
ok_count = 0.
num_class = 3
count = np.zeros(num_class ** 2)
ROWS = 224
COLS = 224
CHANNELS = 3

bark_img_list = []
leaf_img_list = []
form_img_list = []

class_num = []

result_num = 0

#結果の出力ファイル名
f = open('result_form.txt', 'w')
#モデルファイルと重みの読み込み

model = load_model('./my_model_RGB_D.h5')
model.load_weights('./my_model_weights_RGB_D.h5')

#RGB画像
test_dir = "C:/Users/hash/PycharmProjects/keras_deep/RGB-D/test/"

for test_path in os.listdir(test_dir):
    for dir in os.listdir('{}{}'.format(test_dir, test_path)):
        #print('{}{}/{}'.format(test_dir, test_path, dir))
        for file in os.listdir('{}{}/{}'.format(test_dir, test_path, dir)):
            #print('{}{}/{}'.format(test_dir, test_path, dir))
            if str(test_path) == "depth":
                img = img_to_array(
                    load_img('{}{}/{}/{}'.format(test_dir, test_path, dir, file), target_size=(ROWS, COLS, CHANNELS)))
                bark_img_list.append(img)


            elif str(test_path) == "mas":
                img = img_to_array(
                    load_img('{}{}/{}/{}'.format(test_dir, test_path, dir, file), target_size=(ROWS, COLS, CHANNELS)))
                form_img_list.append(img)


            elif str(test_path) == "rgb":
                img = img_to_array(
                    load_img('{}{}/{}/{}'.format(test_dir, test_path, dir, file), target_size=(ROWS, COLS, CHANNELS)))
                leaf_img_list.append(img)



file_num = len(bark_img_list)

bark_img_list = np.array(bark_img_list)
form_img_list = np.array(form_img_list)
leaf_img_list = np.array(leaf_img_list)



#print(len(result))

result = model.predict(([bark_img_list, form_img_list, leaf_img_list]))
#result_2 = model.predict_classes(([bark_img_list, form_img_list, leaf_img_list]))
#print(result)
one_speciy_tree_num = result_num/num_class


cherry = 0
ginkgo = 0
keyaki = 0
ok_count = 0
max_count = 0

class_1_num = len(os.listdir(test_dir+'depth/1'))
class_2_num = len(os.listdir(test_dir+'depth/2')) + class_1_num
class_3_num = len(os.listdir(test_dir+'depth/3')) + class_2_num



for i, name in enumerate(result):
    cherry = name[0]
    ginkgo = name[1]
    keyaki = name[2]

    if cherry > ginkgo and cherry > keyaki:
        print(i+1, "番目の画像はクラス1", cherry)
        if max_count < class_1_num:
            ok_count = ok_count + 1
            #print(ok_count)
        f.writelines('name:{} label:{} result:{}\n'.format(max_count, 1, cherry))
        max_count = max_count + 1

    if ginkgo > cherry and ginkgo > keyaki:
        print(i+1, "番目の画像はクラス2", ginkgo)
        if class_1_num-1 < max_count < class_2_num:
            ok_count = ok_count + 1
            #print(ok_count)
            #print(max_count)
        f.writelines('name:{} label:{} result:{}\n'.format(max_count, 2, cherry))
        max_count = max_count + 1

    if keyaki > cherry and keyaki> ginkgo:
        print(i+1, "番目の画像はクラス3", keyaki)
        if class_2_num-1 < max_count < class_3_num:
            ok_count = ok_count + 1
            print(ok_count)
        f.writelines('name:{} label:{} result:{}\n'.format(max_count, 3, cherry))
        max_count = max_count + 1

f.writelines('正解率:{}%\n'.format(ok_count / class_3_num * 100))
print("正解率: ", ok_count / class_3_num * 100, "%")
f.close()

