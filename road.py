import sys

import cv2
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, merge, Convolution2D, MaxPooling2D,BatchNormalization, UpSampling2D, Dropout, Dense, Flatten, Layer, InputSpec, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.layers.core import Activation
from sklearn.metrics import jaccard_similarity_score
from time import gmtime, strftime

from utils import N_Cls, get_patches, DF
from utils import M, stretch_n, SB, inDir, GS, combined_images, CCCI_index
from config import ISZ, smooth, dice_coef_smooth, batch_size, num_epoch, learning_rate, beta_1, beta_2, \
    epsilon, image_depth, image_size

def predict_road_image(id, model):
    """
    Predicts one image with id, for model, with trs
    Inputs:
    - id: M image id
    - model: model to predict on
    - trs: image threshold on which to make a binary mask (less than =0, more than =1)
    """
    img = combined_images(id, image_size)
    img_streched = stretch_n(img)

    cnv = np.zeros((image_size, image_size, image_depth)).astype(np.float32)
    prd = np.zeros((1, image_size, image_size)).astype(np.float32)
    cnv[:img.shape[0], :img.shape[1], :] = img_streched

    for i in range(0, np.floor(image_size / ISZ).astype(np.int32)):
        line = []
        for j in range(0, np.floor(image_size / ISZ).astype(np.int32)):
            line.append(cnv[i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ])

        # transposes image to feed into predict, where the dimensions are: (num_samples, depth, x, y)
        norm_patches = 2 * np.transpose(line, (0, 3, 1, 2)) - 1  # it also normalises image to [-1,1]
        #print 'norm', norm_patches.shape
        tmp = model.predict(norm_patches, batch_size=2)
        #print 'tmp ', tmp.shape
        for j in range(tmp.shape[0]):
            prd[:, i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ] = tmp[j]

    return prd[:, :img.shape[0], :img.shape[1]] > 0.1

def jaccard_coef_loss(y_true, y_pred):
    return 1 / jaccard_coef(y_true, y_pred)

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

def road_model():
    #optimizer = SGD(lr=0.005, decay=0.0002, momentum=0.9, nesterov=True)
    optimizer = Adam(lr=0.0001, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    inputs = Input((image_depth, ISZ, ISZ))
    conv1 = Convolution2D(1000, 4, 3, activation='relu', border_mode='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = UpSampling2D(size=(2, 2))(pool1)

    # #act = Activation('softmax')(pool1)
    act = Convolution2D(1, 1, 1, activation='sigmoid', border_mode='same')(pool1)
    model = Model(input=inputs, output=act)
    model.compile(optimizer=optimizer, loss=jaccard_coef_loss)
    return model

def small_model():
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
    model.compile(optimizer=Adam(), loss=jaccard_coef_loss, metrics=[jaccard_coef])
    return model

def roads_train():

    model = road_model()
    #model = small_model()
    if len(sys.argv) > 1:
        model.load_weights(sys.argv[1])
    else:
        x_val, y_val = np.load('data/x_tmp_%d.npy' % N_Cls), np.load('data/y_tmp_%d.npy' % N_Cls)
        y_val = y_val[:, 3, :, :]
        y_val = np.expand_dims(y_val, axis=1)
        img = np.load('data/x_trn_%d.npy' % N_Cls)
        msk = np.load('data/y_trn_%d.npy' % N_Cls)
        #msk = msk[:, :, 3]

        x_trn, y_trn = get_patches(img, msk, 3000)
        print "Brisem iz memorije"
        del img
        del msk

        y_trn = y_trn[:, 3, :, :]
        y_trn = np.expand_dims(y_trn, axis=1)
        print "shapeara", y_trn.shape

        print "[roads_train] Training started"

        model.fit(x_trn, y_trn, batch_size=8, nb_epoch=5, verbose=1, shuffle=True, validation_data=(x_val, y_val))

        model.save_weights('road_weights/road_model'+strftime("%H-%M", gmtime()))

    enumerated_list = enumerate(sorted(DF.ImageId.unique()))

    large_img = np.zeros((1, image_size*5, image_size*5)).astype(np.float32)
    print large_img.shape
    for i, id in enumerated_list:
        msk = predict_road_image(id, model)
        #cv2.imwrite("views/roads/roads-predict-"+str(i)+".png", np.transpose(msk, (1, 2, 0))*255)
        #print "road_img_shape", msk.shape
        large_img[:, i / 5 * image_size:i / 5 * image_size + image_size,i % 5 * image_size:(i % 5) * image_size + image_size] = msk
    print large_img.shape
    print large_img
    cv2.imwrite("views/roads/roads-predict.png", np.transpose(large_img,(1,2,0))*255)


if __name__ == '__main__':
    roads_train()
