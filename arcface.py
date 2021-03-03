from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import regularizers
import tensorflow as tf

from tensorflow.python.keras import models as km
from utils_data import loadmat_1, acc_calc, loadmat_2
import utils_network as net
import numpy as np

class ArcFace(Layer):
    """
    Ref: ArcFace：Additive Angular Margin Loss for Deep Face Recognition
    """
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[0][-1].value, self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)
        return out

    def compute_output_shape(self, input_shape):
        return None, self.n_classes


if __name__ == '__main__':
    # data_train, label_train, data_test, label_test = loadmat_2('D:/A_/Enose_datasets/5board/board_lite.mat',
    #                                                            shuffle=True, split=0.9)
    data_train, label_train, data_test, label_test = loadmat_1('D:/A_/Enose_datasets/10board/Batch.mat',
                                                               shuffle=True, split=0.9, batch=10)

    '''Train Network & Save Model'''
    model = net.network_1a_arcface()
    history = model.fit([data_train, label_train], label_train, batch_size=80, epochs=30, verbose=1,
                        callbacks=None, validation_split=0.0, validation_data=[[data_test, label_test], label_test], shuffle=True,
                        class_weight=None, sample_weight=None, initial_epoch=0)
    # model.save('./dcnn_m.h5')

    '''Load model from file & Predict'''
    # model = km.load_model('./h5bkp/dcnn_2.h5')
    # model.summary()
    result = model.predict([data_test, label_test])
    acc = acc_calc(label_test, result)
    print("\n=================validation=================\n")
    print("总精度:{}\n气体1上的精度：{}\n气体2上的精度：{}\n气体3上的精度：{}".format(acc[6],acc[0],acc[1],acc[2]))
    print("气体4上的精度：{}\n气体5上的精度：{}\n气体6上的精度：{}".format(acc[3],acc[4],acc[5]))
