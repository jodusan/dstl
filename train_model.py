import sys

import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, BatchNormalization, UpSampling2D, Dropout, Merge
from keras.models import Model, Sequential
from keras.optimizers import Adam
from sklearn.metrics import jaccard_similarity_score
import theano.tensor as T

from utils import get_patches, get_class_patches
from config import ISZ, smooth, dice_coef_smooth, batch_size, num_epoch, train_patches, learning_rate, beta_1, beta_2, \
    epsilon, image_depth, N_Cls

optimizer = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)


def default_multi_model():
    main_model = MultiModel()
    for i in range(10):
        if i == 3:
            main_model.mm_append(get_small_unet(jaccard_coef_loss))
        else:
            main_model.mm_append(get_unet('binary_crossentropy'))
    return main_model


def train_net():
    """
    Loads train and validation data, gets patches and fits a model.
    Returns:
        - model: trained model
    """
    x_val, y_val = np.load('data/x_tmp_%d.npy' % N_Cls), np.load('data/y_tmp_%d.npy' % N_Cls)
    img = np.load('data/x_trn_%d.npy' % N_Cls)
    msk = np.load('data/y_trn_%d.npy' % N_Cls)

    # x_trn, y_trn = get_patches(img, msk, amt=train_patches)
    x_trn, y_trn = get_class_patches(3, img, msk, max_amount=3000)

    main_model = default_multi_model()

    if len(sys.argv) > 1:
        main_model.mm_load_weights(sys.argv[1])

    # inputs = []
    # labels = []
    # x_valid = []
    # y_valid = []
    # for i in range(10):
    #     inputs.append(x_trn)
    #     labels.append([y_trn[:, np.newaxis, i]])
    #     x_valid.append(x_val)
    #     y_valid.append([y_val[:, np.newaxis, i]])

    main_model.mm_set_model(3, get_small_unet(jaccard_coef_loss))
    # main_model.mm_fit(inputs, labels, x_valid, y_valid)
    main_model.mm_fit_one(3, x_trn, y_trn)

    calc_jacc(main_model, img=x_val, msk=y_val)
    return main_model
    # model_checkpoint = ModelCheckpoint('weights/unet_tmp.hdf5', monitor='loss', save_best_only=True)
    # for i in range(1):
    #     model.fit(inputs, y_trn, batch_size=batch_size, nb_epoch=num_epoch, verbose=1,
    #               shuffle=True,
    #               callbacks=[model_checkpoint], validation_data=(val_inputs, y_val))
    #     del x_trn
    #     del y_trn
    #     score, trs = calc_jacc(model, x_val, y_val)
    #     model.save_weights('weights/unet_10_%d_%d_jk%.4f' % (batch_size, num_epoch, score))
    #     # x_trn, y_trn = get_patches(img, msk)
    #
    # return model


def jaccard_coef_loss(y_true, y_pred):
    return -jaccard_coef(y_true, y_pred)


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


def get_small_unet(loss):
    inputs = Input((image_depth, ISZ, ISZ))
    conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', dim_ordering="th")(inputs)
    conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', dim_ordering="th")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), dim_ordering="th")(conv1)

    conv5 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering="th")(pool1)
    conv5 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering="th")(conv5)

    up9 = merge([UpSampling2D(size=(2, 2), dim_ordering="th")(conv5), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', dim_ordering="th")(up9)
    conv9 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', dim_ordering="th")(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid', dim_ordering="th")(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(), loss=loss, metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    return model


def get_unet(loss):
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

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=[jaccard_coef_loss, jaccard_coef_int])
    return model


def calc_jacc(model, img='load', msk='load'):
    """
    Tries to predict image from validation and returns the jacc score
    Inputs:
    - model: the trained model
    Returns:
    - score: the average jacc score from all classes
    - trs: class thresholds
    """
    if img == 'load':
        img = np.load('data/x_tmp_%d.npy' % N_Cls)  # Opens validation dataset
    if msk == 'load':
        msk = np.load('data/y_tmp_%d.npy' % N_Cls)

    prd = model.mm_predict(img)
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


class MultiModel:
    def __init__(self):
        self.model_list = []
        self.weights_list = []
        self.output_size_list = []

    def mm_append(self, model):
        self.model_list.append(model)

    def mm_load_weights(self, path):
        for i in range(len(self.model_list)):
            self.model_list[i].load_weights(path + str(i))

    def mm_get_model(self, i):
        return self.model_list[i]

    def mm_set_model(self, i, model):
        self.model_list[i] = model

    def mm_fit_one(self, index, input_data, label_data):
        print "[MultiModel - fit] Training model number: ", index + 1
        model = self.model_list[index]
        model_checkpoint = ModelCheckpoint('weights/unet_tmp_' + str(index) + '.hdf5', monitor='loss',
                                           save_best_only=True)
        model.fit(input_data, label_data, batch_size=batch_size, nb_epoch=num_epoch, verbose=1,
                  shuffle=True,
                  callbacks=[model_checkpoint], validation_split=0.2)
        model.save_weights('weights/multimodel/unet_%d_%d_mn%d' % (batch_size, num_epoch, index))

    def mm_fit(self, input_list, label_list, x_val_list, y_val_list):
        print "[MultiModel - fit] Starting training with: batch size:", batch_size, "optimizer lr:", learning_rate, \
            "model number:", len(self.model_list)
        for i in range(len(self.model_list)):
            print "[MultiModel - fit] Training model number: ", i + 1
            model = self.model_list[i]
            model_checkpoint = ModelCheckpoint('weights/unet_tmp_' + str(i) + '.hdf5', monitor='loss',
                                               save_best_only=True)
            model.fit(input_list[i], label_list[i], batch_size=batch_size, nb_epoch=num_epoch, verbose=1,
                      shuffle=True,
                      callbacks=[model_checkpoint], validation_data=(x_val_list[i], y_val_list[i]))
            model.save_weights('weights/multimodel/unet_%d_%d_mn%d' % (batch_size, num_epoch, i))
        del label_list
        del input_list

    def mm_predict(self, image):
        final_result = np.zeros((image.shape[0], N_Cls, image.shape[2], image.shape[3]))
        previous_depth = 0
        for i in range(len(self.model_list)):
            model = self.model_list[i]
            res = model.predict(image, batch_size=batch_size)
            res_depth = res.shape[1]
            final_result[:, previous_depth:(res_depth + previous_depth)] = res
            previous_depth = res_depth + previous_depth
        return final_result


if __name__ == '__main__':
    train_net()
