import os
import pandas as pd
import random

import cv2
import numpy as np
import tifffile as tiff
from skimage.transform import resize

from config import ISZ, image_size, test_nums, image_scale_min, image_scale_max, N_Cls, image_depth

inDir = 'inputs'
DF = pd.read_csv(inDir + '/train_wkt_v4.csv')
GS = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
SB = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))


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


def A(image_id):
    """
    Opens the tiff image
    Input:
    - image_id: id of the image
    Returns:
    - img: image in the form of HxWx8 if the image is sixteen band
    """
    # __author__ = amaia
    # https://www.kaggle.com/aamaia/dstl-satellite-imagery-feature-detection/rgb-using-m-bands-example
    filename = os.path.join(inDir, 'sixteen_band', '{}_A.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img


def P(image_id):
    """
    Opens the tiff image
    Input:
    - image_id: id of the image
    Returns:
    - img: image in the form of HxWx8 if the image is sixteen band
    """
    # __author__ = amaia
    # https://www.kaggle.com/aamaia/dstl-satellite-imagery-feature-detection/rgb-using-m-bands-example
    filename = os.path.join(inDir, 'sixteen_band', '{}_P.tif'.format(image_id))
    img = tiff.imread(filename)
    return img


def rgb(image_id):
    """
    Opens the tiff image
    Input:
    - image_id: id of the image
    Returns:
    - img: image in the form of HxWx8 if the image is sixteen band
    """
    # __author__ = amaia
    # https://www.kaggle.com/aamaia/dstl-satellite-imagery-feature-detection/rgb-using-m-bands-example
    filename = os.path.join(inDir, 'three_band', '{}.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img


def combined_images(image_id, image_size):
    img_m = M(image_id)
    img_m_resize = cv2.resize(img_m, (image_size, image_size))

    img_a = A(image_id)
    img_a_resize = cv2.resize(img_a, (image_size, image_size))

    img_p = P(image_id)
    img_p_resize = cv2.resize(img_p, (image_size, image_size))

    img_rgb = rgb(image_id)
    img_rgb_resize = cv2.resize(img_rgb, (image_size, image_size))

    image = np.zeros((img_rgb_resize.shape[0], img_rgb_resize.shape[1], 20), 'uint8')
    image[..., 0:3] = img_rgb_resize
    image[..., 3] = img_p_resize
    image[..., 4:12] = img_m_resize
    image[..., 12:21] = img_a_resize
    return image


def stretch_n(bands, lower_percent=2, higher_percent=98):
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
    out = np.zeros_like(bands).astype(np.float32)
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

    x, y = [], []

    # Threshold for every of 10 classes TODO: promeniti na klasama koje lose predvidjamo
    tr = [0.4, 0.1, 0.1, 0.15, 0.3, 0.95, 0.1, 0.05, 0.001, 0.005]

    for i in range(amt):

        r_width = random.randint(np.floor(ISZ * image_scale_min), np.floor(ISZ * image_scale_max))
        r_heigth = random.randint(np.floor(ISZ * image_scale_min), np.floor(ISZ * image_scale_max))

        xm = img.shape[0] - r_width
        ym = img.shape[1] - r_heigth

        bad_coords = True
        bad_count = 0

        while bad_coords:
            xc = random.randint(0, xm)  # Get random upper left corner of square patch
            yc = random.randint(0, ym)  # x and y values

            # Exclude list for testing
            exclude = test_nums
            if len(exclude) == 0:
                bad_coords = False
            for excl in exclude:
                if excl / 5 * image_size - r_width <= xc <= excl / 5 * image_size + image_size and excl % 5 * image_size - r_heigth <= yc <= (
                            excl % 5) * image_size + image_size:
                    bad_coords = True
                    bad_count += 1
                    break
                else:
                    bad_coords = False

        im = img[xc:xc + r_width, yc:yc + r_heigth]  # Get square patch starting from xc, yc
        ms = msk[xc:xc + r_width, yc:yc + r_heigth]  # with length of is2
        im = cv2.resize(im, (ISZ, ISZ))  # Resize image
        ms = cv2.resize(ms, (ISZ, ISZ))

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
                    rot = random.randint(1, 5)
                    im = np.rot90(im, rot, (0, 1))
                    ms = np.rot90(ms, rot, (0, 1))

                x.append(im)
                y.append(ms)
                # TODO: add break because it adds unnecessarily many times the same im and ms

    x, y = 2 * np.transpose(x, (0, 3, 1, 2)) - 1, np.transpose(y, (0, 3, 1, 2))
    print "[get_patches] Requested ", amt, " patches. Generated ", x.shape[0], " patches of size ", ISZ, "x", ISZ
    # print x.shape, y.shape, np.amax(x), np.amin(x), np.amax(y), np.amin(y)
    return x, y


def get_class_patches(index, img, msk, offset_x=0, offset_y=0, max_amount=500, validation=0):
    assert max_amount % 2 == 0, "Amount must be a pair number"
    positive_x = []
    negative_x = []
    positive_y = []
    negative_y = []

    x_max = (img.shape[0] - offset_x) / ISZ - 1
    y_max = (img.shape[1] - offset_y) / ISZ - 1

    for i in range(x_max):
        for j in range(y_max):
            ms = msk[i * ISZ + offset_x:(i + 1) * ISZ, j * ISZ + offset_y:(j + 1) * ISZ, np.newaxis, index]
            ms_coord = [i * ISZ + offset_x, (i + 1) * ISZ, j * ISZ + offset_y, (j + 1) * ISZ, index]
            im_coord = [i * ISZ + offset_x, (i + 1) * ISZ, j * ISZ + offset_y, (j + 1) * ISZ]

            if 1 in ms:
                positive_x.append(im_coord)
                positive_y.append(ms_coord)
            else:
                negative_x.append(im_coord)
                negative_y.append(ms_coord)

    # - x: images of shape (N, num channels, ISZ, ISZ)
    # - y: masks of shape (N, 1, ISZ, ISZ)
    #  - img: images of shape (W, H, num channels) (usually 4175, 4175, 8)
    #  - msk: label masks of shape (W, H, num classes) (usually 4175, 4175, 10)
    print "[get_class_patches] Found", len(positive_x), "positive samples and", len(negative_x), "negative samples"

    if len(positive_x) + len(negative_x) >= max_amount:  # more samples than max samples
        element_positions_pos = random.sample(range(0, len(positive_x)), min([max_amount / 2, positive_x]))
        element_positions_neg = random.sample(range(0, len(negative_x)), min([max_amount / 2, negative_x]))
    else:  # if its less get all samples
        element_positions_pos = range(0, len(positive_x))
        element_positions_neg = range(0, len(negative_x))

    num_elements = len(element_positions_pos) + len(element_positions_neg)

    pos_val_ind = []
    neg_val_ind = []
    num_elements_val = 0
    if validation > 0:
        pos_val_ind = random.sample(element_positions_pos,
                                    np.floor(len(element_positions_pos) * validation).astype(np.int32))
        neg_val_ind = random.sample(element_positions_neg,
                                    np.floor(len(element_positions_neg) * validation).astype(np.int32))
        num_elements = num_elements - len(pos_val_ind) - len(neg_val_ind)
        num_elements_val = len(pos_val_ind) + len(neg_val_ind)

    x = np.empty((num_elements, image_depth, ISZ, ISZ))
    y = np.empty((num_elements, 1, ISZ, ISZ))
    val_x = np.empty((num_elements_val, image_depth, ISZ, ISZ))
    val_y = np.empty((num_elements_val, 1, ISZ, ISZ))

    indices = random.sample(range(0, x.shape[0]), x.shape[0])
    val_indices = random.sample(range(0, val_x.shape[0]), val_x.shape[0])

    set_iter = 0
    set_iter_val = 0

    for i in element_positions_pos:
        cx = positive_x[i]
        cy = positive_y[i]
        if i in pos_val_ind:
            val_x[val_indices[set_iter_val]] = 2 * np.transpose(img[cx[0]:cx[1], cx[2]:cx[3]], (2, 0, 1)) - 1
            val_y[val_indices[set_iter_val]] = np.transpose(msk[cy[0]:cy[1], cy[2]:cx[3], np.newaxis, cy[4]], (2, 0, 1))
            set_iter_val += 1
        else:
            x[indices[set_iter]] = 2 * np.transpose(img[cx[0]:cx[1], cx[2]:cx[3]], (2, 0, 1)) - 1
            y[indices[set_iter]] = np.transpose(msk[cy[0]:cy[1], cy[2]:cx[3], np.newaxis, cy[4]], (2, 0, 1))
            set_iter += 1

    for i in element_positions_neg:
        cx = negative_x[i]
        cy = negative_y[i]
        if i in neg_val_ind:
            val_x[val_indices[set_iter_val]] = 2 * np.transpose(img[cx[0]:cx[1], cx[2]:cx[3]], (2, 0, 1)) - 1
            val_y[val_indices[set_iter_val]] = np.transpose(msk[cy[0]:cy[1], cy[2]:cx[3], np.newaxis, cy[4]], (2, 0, 1))
            set_iter_val += 1
        else:
            x[indices[set_iter]] = 2 * np.transpose(img[cx[0]:cx[1], cx[2]:cx[3]], (2, 0, 1)) - 1
            y[indices[set_iter]] = np.transpose(msk[cy[0]:cy[1], cy[2]:cx[3], np.newaxis, cy[4]], (2, 0, 1))
            set_iter += 1

    if validation > 0:
        print "[get_class_patches] got train patches of shape:", x.shape, "and val patches of shape:", val_x.shape
        return x, y, val_x, val_y
    print "[get_class_patches] got train patches of shape:", x.shape
    return x, y


def augment_dataset(img, msk):
    """
    Augment dataset by 8 times
    Args:
        img: image samples to augment (image_samples, num_channels, ISZ, ISZ)
        msk: mask samples to augment (mask_samples, num_classes, ISZ, ISZ)
    Returns:
        x: 8*img_samples, augmented by all rotations and flips
        y: 8*msk_samples, augmented by all rotations and flips
    """
    x = np.empty((img.shape[0] * 8, img.shape[1], img.shape[2], img.shape[3]))
    y = np.empty((msk.shape[0] * 8, msk.shape[1], msk.shape[2], msk.shape[3]))

    indices = random.sample(range(0, x.shape[0]), x.shape[0])
    ind = 0
    for i in range(img.shape[0]):
        x[indices[ind]] = img[i]
        y[indices[ind]] = msk[i]
        ind += 1
        x[indices[ind]] = np.rot90(img[i], 1, (1, 2))
        y[indices[ind]] = np.rot90(msk[i], 1, (1, 2))
        ind += 1
        x[indices[ind]] = np.rot90(img[i], 2, (1, 2))
        y[indices[ind]] = np.rot90(msk[i], 2, (1, 2))
        ind += 1
        x[indices[ind]] = np.rot90(img[i], 3, (1, 2))
        y[indices[ind]] = np.rot90(msk[i], 3, (1, 2))
        ind += 1
        flipped = np.fliplr(img[i])
        flipped_y = np.fliplr(msk[i])
        x[indices[ind]] = flipped
        y[indices[ind]] = flipped_y
        ind += 1
        x[indices[ind]] = np.rot90(flipped, 1, (1, 2))
        y[indices[ind]] = np.rot90(flipped_y, 1, (1, 2))
        ind += 1
        x[indices[ind]] = np.rot90(flipped, 2, (1, 2))
        y[indices[ind]] = np.rot90(flipped_y, 2, (1, 2))
        ind += 1
        x[indices[ind]] = np.rot90(flipped, 3, (1, 2))
        y[indices[ind]] = np.rot90(flipped_y, 3, (1, 2))
        ind += 1
    return x, y


def CCCI_index(id):
    rgb_image = tiff.imread(inDir + '/three_band/{}.tif'.format(id))
    rgb_image = np.rollaxis(rgb_image, 0, 3)
    m = tiff.imread(inDir + '/sixteen_band/{}_M.tif'.format(id))

    RE = resize(m[5, :, :], (rgb_image.shape[0], rgb_image.shape[1]))
    MIR = resize(m[7, :, :], (rgb_image.shape[0], rgb_image.shape[1]))
    R = rgb_image[:, :, 0]
    # canopy chloropyll content index
    CCCI = (MIR - RE) / (MIR + RE) * (MIR - R) / (MIR + R)
    return resize(CCCI, (image_size, image_size))


# def ccci_index(img):
#     m_image = img[..., 4:12]
#     rgb_image = img[..., 0:3]
#     re = m_image[:, :, 5]
#     mir = m_image[:, :, 7]
#     r = rgb_image[:, :, 0]
#     # canopy chlorophyll content index
#     ccci = (mir - re) / (mir + re) * (mir - r) / (mir + r)
#     return ccci


def polygons_to_mask(polygons, im_size):
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
