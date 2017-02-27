import sys

import numpy as np

from config import ISZ, image_size, image_depth, test_nums
from train_model import get_unet, calc_jacc
from utils import N_Cls, stretch_n, combined_images, DF


def make_test_patches(id, model, label):
    img = combined_images(id, image_size)
    x = stretch_n(img)

    cnv = np.zeros((image_size, image_size, image_depth)).astype(np.float32)
    cnv[:img.shape[0], :img.shape[1], :] = x

    # Stack image and label patches
    patch_imgs = []
    patch_labels = []
    for i in range(0, np.floor(image_size / ISZ).astype(np.int32)):
        for j in range(0, np.floor(image_size / ISZ).astype(np.int32)):
            patch_imgs.append(cnv[i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ])
            patch_labels.append(label[i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ])

    patch_imgs = 2 * np.transpose(patch_imgs, (0, 3, 1, 2)) - 1
    patch_labels = 2 * np.transpose(patch_labels, (0, 3, 1, 2)) - 1

    tmp = model.evaluate(patch_imgs, patch_labels, batch_size=4)
    return tmp


def predict_test_images(model, trs):
    # test_imgs = enumerate(sorted(DF.ImageId.unique())[test_nums])
    labels = np.load('data/test_%d.npy' % N_Cls)
    test_imgs = sorted(DF.ImageId.unique())
    test_imgs = [test_imgs[i] for i in test_nums]
    test_results = []
    for i, id in enumerate(test_imgs):
        print "[predict_test_images] Predicting image #", i, " id", id
        msk = make_test_patches(id, model, labels[i])
        test_results.append(msk)
    
    test_results = np.vstack(test_results)
    for k, val in enumerate(np.mean(test_results, axis=0)):
        print model.metrics_names[k], ": ", val


model = get_unet()
model.load_weights(sys.argv[1])
score, trs = calc_jacc(model)
predict_test_images(model, trs)
