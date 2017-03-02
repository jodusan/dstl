import sys

import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, BatchNormalization, UpSampling2D, Dropout, Merge
from keras.models import Model, Sequential
from keras.optimizers import Adam
from sklearn.metrics import jaccard_similarity_score
import theano.tensor as T

from utils import N_Cls, get_patches
from config import ISZ, smooth, dice_coef_smooth, batch_size, num_epoch, train_patches, learning_rate, beta_1, beta_2, \
    epsilon, image_depth

optimizer = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)


def train_net():
    """
    Loads train and validation data, gets patches and fits a model.
    Returns:
        - model: trained model
    """
    x_val, y_val = np.load('data/x_tmp_%d.npy' % N_Cls), np.load('data/y_tmp_%d.npy' % N_Cls)
    img = np.load('data/x_trn_%d.npy' % N_Cls)
    msk = np.load('data/y_trn_%d.npy' % N_Cls)

    x_trn, y_trn = get_patches(img, msk, amt=train_patches)

    model = get_combined_model()
    if len(sys.argv) > 1:
        model.load_weights(sys.argv[1])

    print "[train_net] Training started with: batch size:", batch_size, "optimizer lr:", learning_rate
    model_checkpoint = ModelCheckpoint('weights/unet_tmp.hdf5', monitor='loss', save_best_only=True)
    for i in range(1):
        model.fit([x_trn, x_trn], y_trn, batch_size=batch_size, nb_epoch=num_epoch, verbose=1,
                  shuffle=True,
                  callbacks=[model_checkpoint], validation_data=([x_val, x_val], y_val))
        del x_trn
        del y_trn
        score, trs = calc_jacc(model, x_val, y_val)
        model.save_weights('weights/unet_10_%d_%d_jk%.4f' % (batch_size, num_epoch, score))
        # x_trn, y_trn = get_patches(img, msk)

    return model


def jaccard_coef_loss(y_true, y_pred):
    return 1 / jaccard_coef(y_true, y_pred)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + dice_coef_smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + dice_coef_smooth)


def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    """

    Input:
    - y_true: true labels, theano/tensorflow tensor
    - y_pred: predictions, theano/tensorflow tensor of same shape as y_true
    Return: single tensor value representing the mean of the ouput array across all datapoints
    -
    """
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def third_network():
    inputs = Input((image_depth, ISZ, ISZ))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    up6 = Dropout(0.5)(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    up7 = Dropout(0.5)(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    up8 = Dropout(0.5)(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    up9 = Dropout(0.5)(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(6, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=[jaccard_coef_loss, jaccard_coef_int])
    return model


def simple_road_model():
    inputs = Input((image_depth, ISZ, ISZ))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    up6 = Dropout(0.5)(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    up7 = Dropout(0.5)(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    up8 = Dropout(0.5)(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    up9 = Dropout(0.5)(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(5, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=[jaccard_coef_loss, jaccard_coef_int])
    return model
    # inputs = Input((image_depth, ISZ, ISZ))
    # conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(inputs)
    # pool1 = MaxPooling2D(pool_size=(3, 3), border_mode='same')(conv1)
    # conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool1)
    #
    # conv3 = Convolution2D(2, 1, 1, activation='sigmoid')(conv2)  # Two types of road
    #
    # model = Model(input=inputs, output=conv3)
    # model.compile(optimizer=optimizer, loss='binary_crossentropy',
    #               metrics=[jaccard_coef_loss, jaccard_coef_int, dice_coef_loss])
    # return model


def get_unet():
    inputs = Input((image_depth, ISZ, ISZ))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    up6 = Dropout(0.5)(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    up7 = Dropout(0.5)(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    up8 = Dropout(0.5)(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    up9 = Dropout(0.5)(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(5, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=[jaccard_coef_loss, jaccard_coef_int])
    return model


def get_combined_model():
    unet = get_unet()
    road = simple_road_model()
    third = third_network()

    def f(x):
        a = x[0]
        b = x[1]
        r = T.set_subtensor(a[:, 2:4], b)
        return r[0]

    def f_output(x):
        print "XXXX", x
        return x

    combined_layer = Merge([unet, road], mode='concat', concat_axis=1)

    combined_model = Sequential()
    combined_model.add(combined_layer)
    combined_model.compile(optimizer=optimizer, loss='binary_crossentropy',
                           metrics=[jaccard_coef_loss, jaccard_coef_int])
    return combined_model


def calc_jacc(model, img, msk):
    """
    Tries to predict image from validation and returns the jacc score
    Inputs:
    - model: the trained model
    Returns:
    - score: the average jacc score from all classes
    - trs: class thresholds
    """
    # img = np.load('data/x_tmp_%d.npy' % N_Cls)  # Opens validation dataset
    # msk = np.load('data/y_tmp_%d.npy' % N_Cls)

    print img.shape
    prd = model.predict([img, img], batch_size=batch_size)
    print prd.shape, msk.shape
    avg, trs = [], []

    for i in range(N_Cls):
        t_msk = msk[:, i, :, :]
        t_prd = prd[:, i, :, :]
        # t_msk = t_msk.reshape(msk.shape[0] * msk.shape[2], msk.shape[3])
        # t_prd = t_prd.reshape(msk.shape[0] * msk.shape[2], msk.shape[3])

        m, b_tr = 0, 0
        for j in range(100):
            tr = j / 100.0
            pred_binary_mask = t_prd > tr
            print t_msk.shape
            jk = calc_jacc_numpy(t_msk, pred_binary_mask)
            if jk > m:
                m = jk
                b_tr = tr
        print i, m, b_tr
        avg.append(m)
        trs.append(b_tr)

    score = sum(avg) / 10.0
    return score, trs


def jaccard_coef_int(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    """
    Same as jaccard_coef but clips values to integers.
    Input:
    - y_true: true labels, theano/tensorflow tensor
    - y_pred: predictions, theano/tensorflow tensor of same shape as y_true
    Return: single tensor value representing the mean of the ouput array across all datapoints
    -
    """
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def calc_jacc_numpy(y_true, y_pred):
    pred_binary = y_pred
    msk = y_true

    intersection = np.sum(msk * pred_binary, axis=(0, -1, -2))
    sum_ = np.sum(msk + pred_binary, axis=(0, -1, -2))
    jrk = (intersection + smooth) / (sum_ - intersection + smooth)

    return np.mean(jrk)


if __name__ == '__main__':
    train_net()
