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

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
path_image_test = 'D:\ssx_reasearch\dataset\\test_image.txt'
path_image_train = 'D:\ssx_reasearch\dataset\\train_image.txt'


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
    #print(data.shape)
    #print(labels.shape)
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
    train_data, train_lable = load_image_data(path_image_train)
    test_data, test_lable = load_image_data(path_image_test)
    #train_data, test_data = (train_data.astype('float16') / 255.0).astype('float16'), (test_data.astype('float16') / 255.0).astype('float16')
    train_data, test_data = train_data / 255.0, test_data / 255.0
    model = LeNet5()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])

    checkpoint_save_path = ".\checkpoint\\train_image_2Dcnn.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)

    history = model.fit(train_data, train_lable, batch_size=128, epochs=100, validation_data=(test_data, test_lable),
                        validation_freq=1,
                        callbacks=[cp_callback])
    model.summary()

    ###############################################    save   ###############################################

    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    np.savetxt('.\output\\history_loss.txt', loss)
    np.savetxt('.\output\\history_valloss.txt', val_loss)
    np.savetxt('.\output\\history_acc.txt', acc)
    np.savetxt('.\output\\history_valacc.txt', val_acc)

    ###############################################    show   ###############################################

    # 显示训练集和验证集的acc和loss曲线

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
