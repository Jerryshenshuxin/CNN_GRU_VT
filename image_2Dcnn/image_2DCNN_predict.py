#coding:utf-8

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPool3D, Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense, Conv1D, MaxPool1D, GRU
from tensorflow.keras import Model
from keras.regularizers import l2
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns

path_image_test = 'D:\ssx_reasearch\dataset\\test_image.txt'
checkpoint_save_path='D:\ssx_reasearch\\train\\image_2Dcnn\\6\\checkpoint\\train_image_2Dcnn.ckpt'
target_names = ['卷纸', '硬海绵红圆柱', '塑料红圆柱', '木制红方块', '软泡沫蓝圆柱', '软泡沫蓝高方块', '塑料蓝高方块', '硬海绵红方块','软泡沫蓝扁方块', '蓝硬纸盒', '铁制蓝扁方块', '硬泡沫蓝扁方块', '硬海绵球', '充气球',]
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_image_data(path):
    print("[INFO] loading datas...")
    data = []
    labels = []
    data_paths = []
    f = open(path, 'r')
    lines = f.readlines()
    for line in lines:
        d_path_temp, l_temp = line.split()
        data_paths.append(d_path_temp)
        labels.append(int(l_temp))
    labels = np.array(labels)
    f.close()
    for data_path in data_paths:
        print(data_path)
        image = cv2.imread(data_path)
        image = cv2.resize(image,(160,120))
        data.append(image)
    data = np.array(data)
    return data, labels


weight_decay = 0.01

class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = Conv2D(filters=3, kernel_size=(3, 3),
                         activation='relu',kernel_regularizer=l2(weight_decay))

        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2)

        self.c2 = Conv2D(filters=6, kernel_size=(3, 3),
                         activation='relu',kernel_regularizer=l2(weight_decay))

        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2)

        self.flatten = Flatten()
        self.f1 = Dense(40, activation='sigmoid')
        self.d2 = Dropout(0.5)

        self.f3 = Dense(14, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.p2(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)

        y = self.f3(x)
        return y

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    test_data, test_lable = load_image_data(path_image_test)
    test_data = test_data / 255.0
    model = LeNet5()
    model.load_weights(checkpoint_save_path)
    start = time.clock()
    y=model.predict(test_data)
    end = time.clock()
    print(end-start)
    y_pre=np.argmax(y,axis=1)
    cm=confusion_matrix(test_lable, y_pre)
    #print(cm)
    rs=0
    for i in range(14):
        rs=rs+cm[i][i]
    acc=rs.astype('float')/len(test_lable)
    print(acc)
    cm_f=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    cm_f=(cm_f)
    #print(cm_f)
    np.savetxt('.\confusion_matrix.txt', cm_f)
    ###############################################    show   ###############################################
    '''
    sns.set()
    f, ax = plt.subplots()
    sns.heatmap(cm_f, annot=True,cmap='Greys', ax=ax)  # 画热力图
    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')
    plt.legend()
    plt.show()
    '''
    tick_marks = np.arange(14)
    plt.figure(1, edgecolor='k')
    ax = plt.axes()
    ax.grid(False)
    plt.xticks(tick_marks, target_names, rotation=90, fontsize=8)
    plt.yticks(tick_marks, target_names, rotation=0, fontsize=8)
    plt.ylabel('true')
    plt.xlabel('predict')
    np.set_printoptions(precision=2)
    ind_array = np.arange(14)
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_f[y_val][x_val]
        if (c >= 0.5):
            plt.text(x_val, y_val, "%0.2f" % (c,), color='white', fontsize=7, va='center', ha='center')
        else:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=7, va='center', ha='center')

    plt.imshow(cm_f, interpolation='nearest', cmap=plt.cm.Greys)
    plt.tight_layout()
    plt.savefig('.\confusion_matrix.jpg')
    plt.legend()
    plt.show()