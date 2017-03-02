import sys
import cv2
import numpy as np
from utils import DF
from config import image_size
from train_model import get_combined_model, calc_jacc
from predict_and_submit import predict_image


def predict_train_images(model, trs):
    enumerated_list = enumerate(sorted(DF.ImageId.unique()))
    large_img = np.zeros((10, image_size * 5, image_size * 5)).astype(np.float32)
    for i, id in enumerated_list:
        msk = predict_image(id, model, trs)
        large_img[:, i / 5 * image_size:i / 5 * image_size + image_size,
        i % 5 * image_size:(i % 5) * image_size + image_size] = msk
        print i, id
    for f in range(10):
        cv2.imwrite("views/predict_test_images/ccci/" + str(f + 1) + ".png", large_img[f, :, :] * 255)


if __name__ == '__main__':
    model = get_combined_model()
    model.load_weights(sys.argv[1])
    score, trs = calc_jacc(model)
    predict_train_images(model, trs)
