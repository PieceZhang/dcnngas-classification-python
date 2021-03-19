"""https://github.com/4uiiurz1/keras-arcface"""
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import regularizers
import tensorflow as tf

from tensorflow.python.keras import models as km
from utils_data import loadmat_1, acc_calc, loadmat_2
# from utils_network import network_1a_arcface
import numpy as np


# class ArcFace(Layer):
#     """
#     Ref: ArcFace：Additive Angular Margin Loss for Deep Face Recognition
#     """
#
#     def __init__(self, n_classes, s, m, regularizer=None, **kwargs):
#         super(ArcFace, self).__init__(**kwargs)
#         self.n_classes = n_classes
#         self.s = s
#         self.m = m
#         self.regularizer = regularizers.get(regularizer)
#
#     def build(self, input_shape):
#         super(ArcFace, self).build(input_shape[0])
#         self.W = self.add_weight(name='W',
#                                  shape=(input_shape[0][-1].value, self.n_classes),
#                                  initializer='glorot_uniform',
#                                  trainable=True,
#                                  regularizer=self.regularizer)
#
#     def call(self, inputs):
#         x, y = inputs
#         c = K.shape(x)[-1]
#         # normalize feature
#         x = tf.nn.l2_normalize(x, axis=1)
#         # normalize weights
#         W = tf.nn.l2_normalize(self.W, axis=0)
#         # dot product
#         logits = x @ W
#         # add margin
#         # clip logits to prevent zero division when backward
#         theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
#         target_logits = tf.cos(theta + self.m)
#         # sin = tf.sqrt(1 - logits**2)
#         # cos_m = tf.cos(logits)
#         # sin_m = tf.sin(logits)
#         # target_logits = logits * cos_m - sin * sin_m
#         #
#         logits = logits * (1 - y) + target_logits * y
#         # feature re-scale
#         logits *= self.s
#         out = tf.nn.softmax(logits)
#         return out
#
#     def compute_output_shape(self, input_shape):
#         return None, self.n_classes


class ArcFace(Layer):
    """
    Ref: ArcFace：Additive Angular Margin Loss for Deep Face Recognition
    """

    def __init__(self, n_classes, s, m, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[-1].value, self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 regularizer=self.regularizer)

    def call(self, inputs):
        x = inputs
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        cosa = x @ W
        # clip logits to prevent zero division when backward
        a = tf.acos(K.clip(cosa, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        # softmax
        arcsoftmax = tf.exp(self.s * tf.cos(a + self.m)) / (tf.reduce_sum(tf.exp(self.s * cosa), 1, keepdims=True) -
                                                            tf.exp(self.s * cosa) + tf.exp(self.s * tf.cos(a + self.m)))
        return arcsoftmax

    def compute_output_shape(self, input_shape):
        return None, self.n_classes


class ArcFace2(Layer):
    """
    Ref: ArcFace：Additive Angular Margin Loss for Deep Face Recognition
    """

    def __init__(self, n_classes, s, m, regularizer=None, **kwargs):
        super(ArcFace2, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace2, self).build(input_shape[0])
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


class ArcFaceTrainable(Layer):
    """
    Ref: ArcFace：Additive Angular Margin Loss for Deep Face Recognition
    W,s,m = model.get_layer('...').get_weights()
    s/m可训练会使val_acc产生较大震荡
    """

    def __init__(self, n_classes, regularizer=None, **kwargs):
        super(ArcFaceTrainable, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFaceTrainable, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[-1].value, self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 regularizer=self.regularizer)
        # self.s = self.add_weight(name='s', shape=(1, 1), trainable=True)
        self.s = 4
        # self.m = self.add_weight(name='m', shape=(1, 1), trainable=True)
        self.m = 0.5

    def call(self, inputs):
        x = inputs
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        cosa = x @ W
        # clip logits to prevent zero division when backward
        a = tf.acos(K.clip(cosa, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        # softmax
        arcsoftmax = tf.exp(self.s * tf.cos(a + self.m)) / (tf.reduce_sum(tf.exp(self.s * cosa), 1, keepdims=True) -
                                                            tf.exp(self.s * cosa) + tf.exp(self.s * tf.cos(a + self.m)))
        return arcsoftmax

    def compute_output_shape(self, input_shape):
        return None, self.n_classes


class CosFace(Layer):
    def __init__(self, n_classes, s, m, regularizer=None, **kwargs):
        super(CosFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(CosFace, self).build(input_shape[0])
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
        target_logits = logits - self.m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)


if __name__ == '__main__':
    pass
