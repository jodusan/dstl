import os
import pandas as pd
import sys
from collections import defaultdict

import cv2
import numpy as np
import shapely.affinity
import shapely.wkt
from shapely.geometry import MultiPolygon, Polygon

from utils import N_Cls, M, stretch_n, SB, inDir, GS, combined_images
from config import ISZ, image_size, image_depth
from train_model import get_unet, calc_jacc


def predict_image(id, model, trs):
    """
    Predicts one image with id, for model, with trs
    Inputs:
    - id: M image id
    - model: model to predict on
    - trs: image threshold on which to make a binary mask (less than =0, more than =1)
    """
    img = combined_images(id, image_size)
    x = stretch_n(img)

    cnv = np.zeros((image_size+125, image_size+125, image_depth)).astype(np.float32)
    prd = np.zeros((N_Cls, image_size+125, image_size+125)).astype(np.float32)
    cnv[:img.shape[0], :img.shape[1], :] = x

    for i in range(0, 6):
        line = []
        for j in range(0, 6):
            line.append(cnv[i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ])

        x = 2 * np.transpose(line, (0, 3, 1, 2)) - 1
        tmp = model.predict(x, batch_size=4)
        for j in range(tmp.shape[0]):
            prd[:, i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ] = tmp[j]

    # trs = [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1]
    for i in range(N_Cls):
        prd[i] = prd[i] > trs[i]

    return prd[:, :img.shape[0], :img.shape[1]]


def predict_test_images(model, trs):
    enumerated_list = enumerate(sorted(set(SB['ImageId'].tolist())))
    for i, id in enumerated_list:
        msk = predict_image(id, model, trs)
        np.save('msk/10_%s' % id, msk)
        if i % 100 == 0: print i, id


def mask_to_polygons(mask, epsilon=5, min_area=1.):
    """
    Pravi (multi)poligone od output slike mreze
    Input:
    - mask: mask image
    - epsilon: margin of error
    - min_area: minimal area for polygon

    Returns:
    - all_polygons: all polygons found in mask
    """
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly

    # first, find contours with cv2: it's much faster than shapely
    contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons


def make_submit():
    """
    Make a submission file
    """
    print "[make_submit] Making a submissions file"
    df = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))
    print df.head()
    for idx, row in df.iterrows():
        id = row[0]
        kls = row[1] - 1

        msk = np.load('msk/10_%s.npy' % id)[kls]
        pred_polygons = mask_to_polygons(msk)
        x_max = GS.loc[GS['ImageId'] == id, 'Xmax'].as_matrix()[0]
        y_min = GS.loc[GS['ImageId'] == id, 'Ymin'].as_matrix()[0]

        x_scaler, y_scaler = get_scalers(msk.shape, x_max, y_min)

        scaled_pred_polygons = shapely.affinity.scale(pred_polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler,
                                                      origin=(0, 0, 0))

        df.iloc[idx, 2] = shapely.wkt.dumps(scaled_pred_polygons)
        if idx % 100 == 0: print idx
    print df.head()
    df.to_csv('subm/1.csv', index=False)


def get_scalers(im_size, x_max, y_min):
    """
    Scale image back to xycoordinates
    Inputs:
    - im_size: tuple of image size (h,w)
    - x_max: x max coordinates for image
    - y_min: y min coordinates for image
    Returns:
    - Tuple of values by which to scale image
    """
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly
    h, w = im_size  # they are flipped so that mask_for_polygons works correctly
    h, w = float(h), float(w)
    w_ = 1.0 * w * (w / (w + 1))
    h_ = 1.0 * h * (h / (h + 1))
    return w_ / x_max, h_ / y_min


model = get_unet()
model.load_weights(sys.argv[1])
score, trs = calc_jacc(model)
predict_test_images(model, trs)
make_submit()
