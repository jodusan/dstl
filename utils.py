# from predict_and_submit import calc_jacc, predict_id, predict_test, mask_to_polygons, make_submit
# from train_model import train_net, get_unet

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import tifffile as tiff
import os
import random

N_Cls = 10
inDir = 'inputs'
DF = pd.read_csv(inDir + '/train_wkt_v4.csv')
GS = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
SB = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))
ISZ = 160
smooth = 1e-12


def M(image_id):
    """
    Opens the tiff image
    Input:
    - image_id: id of the image
    Returns:
    - img: image in the form of HxWx8 if the image is sixteen band
    """
    # __author__ = amaia
    # https://www.kaggle.com/aamaia/dstl-satellite-imagery-feature-detection/rgb-using-m-bands-example
    filename = os.path.join(inDir, 'sixteen_band', '{}_M.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img


def stretch_n(bands, lower_percent=5, higher_percent=95):
    """
    Rasiri (po vrednostima) svaki band slike kako bi se videlo vise detalja,
    odseca najvisih i najnizih 5% sa default vrednostima
    Input:
    - bands: slika
    - lower_percent: donji percentil
    - higher_percent: gonji percentil
    Returns:
    - out: Rasirena slika, HxWxBands
    """
    out = np.zeros_like(bands)
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 1  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
        # Sacuva bandove slike pre i posle strech_n u folder bands, napraviti folder pre odkomentarisanja!!
        # cv2.imwrite("bands/band"+str(i)+".png", bands[:, :, i]*255)
        # #cv2.imwrite("bands/out"+str(i)+".png", t*255)

    return out.astype(np.float32)


def get_patches(img, msk, amt=10000, aug=True):
    """
    Returns patches of shape (ISZ, ISZ) from the big picture.
    ISZ - side length of square patch
    Input:
        - img: images of shape (W, H, num channels) (usually 4175, 4175, 8)
        - msk: label masks of shape (W, H, num classes) (usually 4175, 4175, 10)
        - amt: integer for how many patches we want
        - aug: boolean on whether to augment by flipping image vertically or horizontaly
    Return:
        - x: images of shape (N, num channels, ISZ, ISZ)
        - y: masks of shape (N, num classes, ISZ, ISZ)
    """
    is2 = int(1.0 * ISZ)
    xm = img.shape[0] - is2
    ym = img.shape[1] - is2

    x, y = [], []

    # Threshold for every of 10 classes
    tr = [0.4, 0.1, 0.1, 0.15, 0.3, 0.95, 0.1, 0.05, 0.001, 0.005]

    for i in range(amt):
        xc = random.randint(0, xm)  # Get random upper left corner of square patch
        yc = random.randint(0, ym)  # x and y values

        im = img[xc:xc + is2, yc:yc + is2]  # Get square patch starting from xc, yc
        ms = msk[xc:xc + is2, yc:yc + is2]  # with length of is2

        # For every class, loop and see if it passes threshold
        for j in range(N_Cls):
            sm = np.sum(ms[:, :, j])
            if 1.0 * sm / is2 ** 2 > tr[j]:
                if aug:
                    # 0.5 chance to flip it horizontal
                    if random.uniform(0, 1) > 0.5:
                        im = im[::-1]
                        ms = ms[::-1]
                    # 0.5 chance to flip it verticaly
                    if random.uniform(0, 1) > 0.5:
                        im = im[:, ::-1]
                        ms = ms[:, ::-1]

                x.append(im)
                y.append(ms)
                # TODO: add break because it adds unnecessarly many times the same im and ms

    x, y = 2 * np.transpose(x, (0, 3, 1, 2)) - 1, np.transpose(y, (0, 3, 1, 2))
    print "[get_patches] Requested ", amt, " patches. Generated ", x.shape[0], " patches of size ", ISZ, "x", ISZ
    # print x.shape, y.shape, np.amax(x), np.amin(x), np.amax(y), np.amin(y)
    return x, y


def mask_for_polygons(polygons, im_size):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly
    """
    Returns integer based mask for given polygons. If no polygons detected returns empty mask.
    Input:
        - polygons: Multipolygon object
        - im_size: (W,H) width and hight of image ( ie. 837, 851)
    Return:
        - img_mask: generated mask for im_size
    """
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

#
# def check_predict(id='6120_2_3'):
#     model = get_unet()
#     model.load_weights('weights/unet_10_jk0.7878')
#
#     msk = predict_id(id, model, [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1])
#     img = M(id)
#
#     plt.figure()
#     ax1 = plt.subplot(131)
#     ax1.set_title('image ID:6120_2_3')
#     ax1.imshow(img[:, :, 5], cmap=plt.get_cmap('gist_ncar'))
#     ax2 = plt.subplot(132)
#     ax2.set_title('predict bldg pixels')
#     ax2.imshow(msk[0], cmap=plt.get_cmap('gray'))
#     ax3 = plt.subplot(133)
#     ax3.set_title('predict bldg polygones')
#     ax3.imshow(mask_for_polygons(mask_to_polygons(msk[0], epsilon=1), img.shape[:2]), cmap=plt.get_cmap('gray'))
#
#     plt.show()


# if __name__ == '__main__':
#     # bonus
#     check_predict()
