from tensorflow.keras.losses import kullback_leibler_divergence as KLD_Loss, \
    categorical_crossentropy as logloss, mean_absolute_error as meanaloss, mean_squared_error as meansloss,\
    sparse_categorical_crossentropy as slogloss
from keras import backend as K
from functools import partial
import tensorflow as tf
import numpy as np
import math
from keras import regularizers


# 定义知识蒸馏损失函数
def kd_loss(y_true, y_pred):
    NUM_class = int(y_pred.shape[-1])
    soft_label = y_true[:, :NUM_class]  # logits
    MAE_loss = meanaloss(soft_label, y_pred)
    return MAE_loss


# MMD第二种实现
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    # n_s=tf.shape(source)[0]
    # n_s=10 if n_s is None else n_s
    # print('n_s:', n_s, tf.shape(source), tf.shape(source)[0])
    # n_t= tf.shape(target)[0]
    # n_t=10 if n_t is None else n_t
    # print('n_t:', n_t, tf.shape(target), tf.shape(target)[0])

    # n_samples =n_s+n_t  # n_samples必须是string or number，不能是tensor
    # print('n_samples:', n_samples)

    n_s=source.shape.as_list()[0]
    n_s = 10 if n_s is None else n_s
    #print('n_s:', n_s, source.shape.as_list(), source.shape.as_list()[0])
    n_t = target.shape.as_list()[0]
    n_t = 10 if n_t is None else n_t
    #print('n_t:', n_t, target.shape.as_list(), target.shape.as_list()[0])
    n_samples = n_s + n_t  # n_samples必须是string or number，不能是tensor
    #print('n_samples:', n_samples)

    total = tf.concat([source, target], axis=0)                                                      #   [None,n]
    total0 = tf.expand_dims(total,axis=0)               #   [1,b,n]
    total1 = tf.expand_dims(total,axis=1)               #   [b,1,n]
    L2_distance = tf.reduce_sum(((total0 - total1) ** 2),axis=2)     #   [b,b,n]=>[b,b]     #   [None,None,n]=>[128,128,1]
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = tf.reduce_sum(L2_distance) / float(n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [tf.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)   #[b,b]


def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    n_s = source.shape.as_list()[0]
    n_s = 10 if n_s is None else n_s
    n_t = target.shape.as_list()[0]
    n_t = 10 if n_t is None else n_t

    XX = tf.reduce_sum(kernels[:n_s, :n_s])/float(n_s**2)
    YY = tf.reduce_sum(kernels[-n_t:, -n_t:])/float(n_t**2)
    XY = tf.reduce_sum(kernels[:n_s, -n_t:])/float(n_s*n_t)
    YX = tf.reduce_sum(kernels[-n_t:, :n_s])/float(n_s*n_t)
    loss = XX + YY - XY - YX
    return loss


def fedprox_loss(mu, global_w):
    def loss(y_true, y_pred, model_parameters):
        labels = y_true
        CE_loss = slogloss(labels, y_pred)
        dis = [K.sum(regularizers.l2()(model_parameters[i] - global_w[i])) for i in range(len(model_parameters))]
        dis = K.sum(dis)
        return CE_loss + mu * dis
    return loss


def cosine_proximity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.sum(y_true * y_pred, axis=-1)


def moon_loss(temperature):
    def loss(y_true, y_pred):
        len_emb = int(y_pred.shape[1])
        zg = y_true[:, :len_emb]
        zpv = y_true[:, len_emb:]
        a = K.exp(cosine_proximity(y_pred, zg)/temperature)
        b = K.exp(cosine_proximity(y_pred, zg)/temperature) + K.exp(cosine_proximity(y_pred, zpv)/temperature)
        return K.log(b/a)
    return loss


def fedproc_loss(global_embeds):
    def loss(y_true, y_pred):
        a = K.exp(cosine_proximity(y_true, y_pred))
        b = None
        for i in range(len(global_embeds)):
            if b is None:
                b = K.exp(cosine_proximity(y_true, global_embeds[i]))
            else:
                b += K.exp(cosine_proximity(y_true, global_embeds[i]))
        return K.log(b/a)
    return loss


def ditto_loss(lambd, global_w):
    def loss(y_true, y_pred, model_parameters):
        labels = y_true
        CE_loss = slogloss(labels, y_pred)
        dis = [K.sum(regularizers.l2()(model_parameters[i] - global_w[i])) for i in range(len(model_parameters))]
        dis = K.sum(dis)
        return CE_loss + lambd * dis
    return loss


# def fedphp_hloss(y_true, y_pred):
#     # FedPhp: EMNIST:64896
#     y_true = tf.reshape(y_true, shape=(-1, 768))  #cifar：2304 YELP:768 EMNIST:18816,6272
#     #MMD_loss = mmd_loss(y_pred,y_true)
#     MSE_loss = meansloss(y_true, y_pred)
#     return MSE_loss


def fedphp_hloss(size):
    def loss(y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, size))
        MSE_loss = meansloss(y_true, y_pred)
        return MSE_loss
    return loss