from shapely.wkt import loads as wkt_loads

import cv2
import numpy as np

from config import validation_patches, image_size, image_depth, generate_label_masks, test_nums, N_Cls
from utils import DF, stretch_n, get_patches, GS, combined_images


def stick_all_train():
    """
    Sticks all training images into one giant image (835*5=4175 x and y dimension),
    does this also to the masks. It saves the results into the the data folder.
    """
    s = image_size  # image size
    print "[stick_all_train] number of images =", 5 * 5, "size of final image:", 5 * s

    x = np.zeros((5 * s, 5 * s, image_depth))  # Train sticked image
    y = np.zeros((5 * s, 5 * s, N_Cls))  # Label masks sticked image
    t = np.zeros((len(test_nums), s, s, N_Cls))
    t_stacked = 0

    ids = sorted(DF.ImageId.unique())  # All image ids
    print len(ids)
    for i in range(5):
        for j in range(5):
            image_id = ids[5 * i + j]

            img = combined_images(image_id, image_size)  # Loads all image bands with image_size size
            img = stretch_n(img)  # Stretches bands
            print img.shape, image_id, np.amax(img), np.amin(img)
            # Gets a location from the sticked image and fills it with the current image
            x[s * i:s * i + s, s * j:s * j + s, :] = img[:s, :s, :]
            for z in range(N_Cls):
                # Gets a location from the sticked mask and fills it with the mask of current image
                y[s * i:s * i + s, s * j:s * j + s, z] =\
                    generate_mask_for_image_and_class((img.shape[0], img.shape[1]), image_id, z + 1)[:s, :s]
            if (5*i+j) in test_nums:
                t[t_stacked] = y[s * i:s * i + s, s * j:s * j + s, :]

    print np.amax(y), np.amin(y)
    if generate_label_masks:
        cv2.imwrite("views/preprocess/concat-5x5.png", x[:, :, 1]*255)
        for i in range(10):
            cv2.imwrite("views/preprocess/maska"+str(i)+".png", y[:, :, i]*255)
    print "Saving image to data"
    np.save('data/x_trn_%d' % N_Cls, x)
    print "Saving labels to data"
    np.save('data/y_trn_%d' % N_Cls, y)
    print "Saving test labels to data"
    np.save('data/test_%d' % N_Cls, t)


def make_val():
    """
    Makes a validation dataset using patches from main image
    """
    print "let's pick some samples for validation"
    img = np.load('data/x_trn_%d.npy' % N_Cls)
    msk = np.load('data/y_trn_%d.npy' % N_Cls)
    x, y = get_patches(img, msk, amt=validation_patches)

    np.save('data/x_tmp_%d' % N_Cls, x)
    np.save('data/y_tmp_%d' % N_Cls, y)


def _convert_coordinates_to_raster(coords, img_size, xymax):
    """
    Pretvara koordinate iz formata 0-1 u format koji odgovara velicini i lokaciji slike
    Input:
    - coords: matrica koordinata
    - img_size: tupl H i W zeljenje velicine
    - xymax: tupl koji sadrzi koordinate slike
    Return:
    - listu koordinata skaliranu po zeljenoj velicini
    """
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    Xmax, Ymax = xymax
    H, W = img_size
    W1 = 1.0 * W * W / (W + 1)
    H1 = 1.0 * H * H / (H + 1)
    xf = W1 / Xmax
    yf = H1 / Ymax
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


def _get_xmax_ymin(grid_sizes_panda, imageId):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    """
    Gets xmax and ymin for a given imageId from grid_sizes.csv(grid_sizes_panda variable) file.
    Input:
    - grid_sizes_panda: pandas object that contains whole grid_sizes.csv file of shape (450,3)
    - imageId: id of image we are getting xmax and ymin

    Return:
    - xmax, ymin: tuple of values for imageId
    """
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)
    return (xmax, ymin)


def _get_polygon_list(wkt_list_pandas, imageId, cType):
    """
    Gets the poligons from training file
    Input:
    - wkt_list_pandas: panda objekat sa otvorenim csv fajlom training seta
    - imageId: id slike
    - tip klase (1-10)
    Return:
    - Lista poligona iz csv-a
    """
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList


def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    """
    Makes a list of exterior and interior polygons
    Input:
    - polygonList: list of multipolygons
    - raster_img_size: velicina slika sa kojima radimo
    - xymax: xymax za trenutnu sliku
    """
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list, interior_list


def _plot_mask_from_contours(raster_img_size, contours, class_value=1):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    """
    Plot mask from polygon contours.
    Input:
    - raster_img_size: size of the raster image we want (ie. 500x500)
    - contours: tuple of (outer_polygons, inner_polygons)
    - class_value: Value 0-255 defining color where 0=white, 255=black
    Return:
    - img_mask: 2d array mask of polygons
    """
    img_mask = np.zeros(raster_img_size, np.uint8)
    if contours is None:
        return img_mask
    perim_list, interior_list = contours
    cv2.fillPoly(img_mask, perim_list, class_value)
    cv2.fillPoly(img_mask, interior_list, 0)
    return img_mask


def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda=GS, wkt_list_pandas=DF):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    """
    Generate image mask for a given class.
    Input:
    - raster_img_size: size of the raster image we want (ie. 500x500)
    - imageId: id of image we want
    - class_type: class integer describing our class
    - grid_sizes_panda: pandas object that contains whole grid_sizes.csv file of shape (450,3)
    - wkt_list_pandas: pandas object that contains whole train_wkt_v4.csv file of shape (250, 3)
    Return:
    - img_mask: 2d array mask of polygons for given class
    """
    xymax = _get_xmax_ymin(grid_sizes_panda, imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas, imageId, class_type)
    contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
    mask = _plot_mask_from_contours(raster_size, contours, 1)
    return mask


if __name__ == '__main__':
    stick_all_train()
    make_val()
