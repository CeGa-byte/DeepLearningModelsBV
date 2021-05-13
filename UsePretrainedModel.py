import tensorflow as tf
import keras
import os
import numpy as np
import albumentations as A
import random
import cv2
import statistics

from PIL import Image
from scipy import ndimage
from skimage.io import imread, imshow
from skimage.transform import resize
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import binarize
import matplotlib.pyplot as plt
from keras import backend as K


def calc_iou(y_true, y_pred):
    """ Calculate Intersection Over Union for binarized prediction and mask """
    return np.sum(np.logical_and(y_true, y_pred)) / np.sum(np.logical_or(y_true, y_pred))


def confusion_matrix(y_true, y_pred):
    """
    Calculates the Metrics of the Confusion Matrix pixelwise: FP, FN, TP, TN
    :param y_pred: Masks predicted by the model
    :param y_true: Real masks
    :return: FP, FN, TP, TN
    """

    TP = np.sum(np.logical_and(y_true, y_pred))
    TN = np.sum(np.logical_and(np.logical_not(y_true), np.logical_not(y_pred)))
    FP = np.sum(np.logical_and(np.logical_not(y_true), y_pred))
    FN = np.sum(np.logical_and(y_true, np.logical_not(y_pred)))
    return FP, FN, TP, TN


def IoU(y_true, y_pred):
    """ Intersection over Union metric """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1))

    # sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    # iou_score = intersection / (sum_ - intersection)
    return intersection / (sum_ - intersection)


def jaccard_distance(y_true, y_pred, smooth=1):
    """ Calculates dissimilarity of sample sets. By subtracting from 1 we get percentage of similarity """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


def dice_coef_loss(y_true, y_pred):
    """ Dice Loss function to maximize """
    return 1 - dice_coef(y_true, y_pred)


def find_threshold(predictions, groundTruth_masks):
    """
    Get threshold for predictions to maximize intersection over union
    :param predictions: predictions from the model with values 0-1
    :param groundTruth_masks: ground truth mask to predict
    :return: threshold value which maximizes intersection over union
    """
    thresholds = []  # Each entry represents best threshold for one prediction
    for i in range(len(predictions)):
        best_iou = 0
        best_threshold = 0
        img = predictions[i]
        for j in range(1, 10):
            current_threshold = j / 10  # threshold in step size 0.1
            thresh_img = (img > current_threshold).astype(np.int)
            iou = calc_iou(groundTruth_masks[i], thresh_img)  # Current IoU for current threshold
            if iou > best_iou:
                best_iou = iou
                best_threshold = current_threshold
        thresholds.append(best_threshold)

    threshold_value = statistics.mean(thresholds)  # Get mean of all thresholds as global threshold

    return threshold_value


def load_images(img_path, max_height, max_width, convert=True):
    """
    Load images which should be segmented, ideally all images are the same size.
    :param img_path: path to the images, real images not the .matlab or .npy files
    :param max_height: maximum height of the images
    :param max_width: maximum width of the images
    :param convert: convert to 0-1 if range is 0-255
    :return:
    """
    image_files = glob(os.path.join(img_path, '*.*'))
    images_list = []
    for _ in range(len(image_files)):
        image = cv2.imread(image_files[_])
        images_list.append(image)
    for _ in range(len(images_list)):
        images_list[_] = make_border(images_list[_], max_height, max_width)

    images_list = make_ndarray(images_list, convert=convert)
    images_list = slice_patches(images_list, PATCH_HEIGHT, PATCH_WIDTH)

    return images_list


def make_border(data, wanted_height: int, wanted_width: int):
    """
    Create a border around an image or mask to make it the desired size if it is smaller!
    :param data: image or mask smaller than wanted
    :param wanted_height: wanted size on the y axis
    :param wanted_width: wanted size on the x axis
    :return: image or mask of desired size with borders to fill extra space
    """
    current_height = data.shape[0]
    current_width = data.shape[1]
    add_sides = (wanted_width - current_width) // 2
    add_top_bottom = (wanted_height - current_height) // 2

    border_img = cv2.copyMakeBorder(data, add_top_bottom, add_top_bottom, add_sides, add_sides, cv2.BORDER_CONSTANT,
                                    value=[0, 0, 0])

    return border_img


def slice_patches(data, wanted_height: int, wanted_width: int):
    """
    Slice data images into smaller patches to feed into model, height and width should be dividable by 16.
    Always adding one additional patch and generating overflow, which is distributed as overlap along all patches. Last
    patch may overlap more, if overlap distributed along all patches is odd, because then overlap is floored.
    :param data: images or masks
    :param wanted_height: height of one patch
    :param wanted_width: width of one patch
    :return: return all patches of the data
    """
    patches = []
    for _ in range(len(data)):
        current_height = data[_].shape[0]
        current_width = data[_].shape[1]

        # If patches fit image perfectly, no overflow handling required
        if PATCHES * wanted_height == current_height and PATCHES * wanted_width == current_width:
            fitting_patches_height = PATCHES
            step_size_height = wanted_height

            fitting_patches_width = PATCHES
            step_size_width = wanted_width

            for nmr_patch_height in range(fitting_patches_height):
                for nmr_patch_width in range(fitting_patches_width):
                    patch = [
                        data[_][i][nmr_patch_width * step_size_width:nmr_patch_width * step_size_width + wanted_width]
                        for i in
                        range(nmr_patch_height * step_size_height, nmr_patch_height * step_size_height + wanted_height)]
                    patches.append(patch)

        # If patches don't fit along height, y-axis
        elif PATCHES * wanted_height > current_height and PATCHES * wanted_width == current_width:
            fitting_patches_height = PATCHES - 1  # Last patch may not fit with the same step size
            overflow_height = PATCHES * wanted_height - current_height
            overlap_height = overflow_height // fitting_patches_height
            step_size_height = wanted_height - overlap_height

            fitting_patches_width = PATCHES
            step_size_width = wanted_width

            for nmr_patch_height in range(fitting_patches_height):
                for nmr_patch_width in range(fitting_patches_width):
                    patch = [
                        data[_][i][nmr_patch_width * step_size_width:nmr_patch_width * step_size_width + wanted_width]
                        for i in
                        range(nmr_patch_height * step_size_height, nmr_patch_height * step_size_height + wanted_height)]
                    patches.append(patch)

            # Patches, which may not fit with same step size may overlap more along y axis
            for nmr_patch_width in range(fitting_patches_width):
                patch = [
                    data[_][i][nmr_patch_width * step_size_width:nmr_patch_width * step_size_width + wanted_width] for i
                    in range(-wanted_height, 0)]
                patches.append(patch)

        # If patches don't fit along width, x axis
        elif PATCHES * wanted_height == current_height and PATCHES * wanted_width > current_width:
            fitting_patches_height = PATCHES
            step_size_height = wanted_height

            fitting_patches_width = PATCHES - 1
            overflow_width = PATCHES * wanted_width - current_width
            overlap_width = overflow_width // fitting_patches_width
            step_size_width = wanted_width - overlap_width

            for nmr_patch_height in range(fitting_patches_height):
                for nmr_patch_width in range(fitting_patches_width):
                    patch = [
                        data[_][i][nmr_patch_width * step_size_width:nmr_patch_width * step_size_width + wanted_width]
                        for i in
                        range(nmr_patch_height * step_size_height, nmr_patch_height * step_size_height + wanted_height)]
                    patches.append(patch)

                # Patch which may not fit with same step size, overlaps more along x axis
                patch = [data[_][i][-wanted_width:] for i in
                         range(nmr_patch_height * step_size_height,
                               nmr_patch_height * step_size_height + wanted_height)]
                patches.append(patch)

        # If patches don't fit along neither height nor width
        elif PATCHES * wanted_height > current_height and PATCHES * wanted_width > current_width:
            fitting_patches_height = PATCHES - 1  # Last patch may not fit with the same step size
            overflow_height = PATCHES * wanted_height - current_height
            overlap_height = overflow_height // fitting_patches_height
            step_size_height = wanted_height - overlap_height

            fitting_patches_width = PATCHES - 1
            overflow_width = PATCHES * wanted_width - current_width
            overlap_width = overflow_width // fitting_patches_width
            step_size_width = wanted_width - overlap_width

            for nmr_patch_height in range(fitting_patches_height):
                for nmr_patch_width in range(fitting_patches_width):
                    patch = [
                        data[_][i][nmr_patch_width * step_size_width:nmr_patch_width * step_size_width + wanted_width]
                        for i in
                        range(nmr_patch_height * step_size_height, nmr_patch_height * step_size_height + wanted_height)]
                    patches.append(patch)

                # Patch which may not fit with same step size, overlaps more
                patch = [data[_][i][-wanted_width:] for i in
                         range(nmr_patch_height * step_size_height,
                               nmr_patch_height * step_size_height + wanted_height)]
                patches.append(patch)

            for nmr_patch_width in range(fitting_patches_width):
                patch = [
                    data[_][i][nmr_patch_width * step_size_width:nmr_patch_width * step_size_width + wanted_width] for i
                    in range(-wanted_height, 0)]
                patches.append(patch)

            patch = [data[_][i][-wanted_width:] for i in
                     range(-wanted_height, 0)]  # Last patch which may not fit neither height nor width
            patches.append(patch)

    return np.array(patches)


def recreate_from_patches(data):
    """
    Recreate images from patches by putting them back together
    :param data: patches from the images
    :return: list of images recreate out of their patches
    """
    overlap_height = (PATCHES * PATCH_HEIGHT - IMG_HEIGHT) // (PATCHES - 1)  # Overlap of patches along y axis
    step_size_height = PATCH_HEIGHT - overlap_height  # Step size along y axis

    overlap_width = (PATCHES * PATCH_WIDTH - IMG_WIDTH) // (PATCHES - 1)  # Overlap of patches along x axis
    step_size_width = PATCH_WIDTH - overlap_width  # Step size along x axis

    whole_images = []
    i = 0
    while i < len(data):
        image = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))  # Create an empty image to pin patches on

        for h in range(PATCHES - 1):
            for w in range(PATCHES - 1):
                # Insert patches into image starting from top left corner, without the patches touching right or bottom border
                if h > 0:  # First row has no overlap with patches above them
                    if overlap_height > 0:
                        # Create array of overlap along y axis with mean values from overlapping patches
                        mean_overlap_height = cv2.addWeighted(data[i][:overlap_height], 0.5,
                                                              data[i - PATCHES][step_size_height:], 0.5, 0)

                        # Insert into patch where it overlaps
                        rest = data[i][overlap_height:]
                        data[i] = np.append(mean_overlap_height, rest, axis=0)

                # Insert patch into image
                image = insert_patch_subpixel(image, data[i], (w * step_size_width + PATCH_WIDTH / 2,
                                                               h * step_size_height + PATCH_HEIGHT / 2))

                if w == PATCHES - 2:  # If we are at the second to last patch, overlap may be calculated different
                    i += 1
                    continue

                else:
                    i += 1
                    if overlap_width > 0:
                        # Create array of overlap with mean values from overlapping patches
                        mean_overlap_width = cv2.addWeighted(data[i][:, [i for i in range(0, overlap_width)]], 0.5,
                                                             data[i - 1][:,
                                                             [i for i in range(PATCH_WIDTH - overlap_width,
                                                                               PATCH_WIDTH)]], 0.5, 0)
                        # Insert into next patch
                        rest = data[i][:, [i for i in range(overlap_width, PATCH_WIDTH)]]
                        data[i] = np.append(mean_overlap_width, rest, axis=1)

            # Insert patch which touches right border on this height, may overlap more
            overlap_last_width = (PATCH_WIDTH + (PATCHES - 2) * step_size_width) - (IMG_WIDTH - PATCH_WIDTH)

            if overlap_last_width > 0:
                # Create array of overlap with mean values from overlapping patches
                mean_overlap_width = cv2.addWeighted(data[i][:, [i for i in range(0, overlap_last_width)]], 0.5,
                                                     data[i - 1][:, [i for i in range(PATCH_WIDTH - overlap_last_width,
                                                                                      PATCH_WIDTH)]], 0.5, 0)
                # Insert array of overlap into patch, where it overlaps
                rest = data[i][:, [i for i in range(overlap_last_width, PATCH_WIDTH)]]
                data[i] = np.append(mean_overlap_width, rest, axis=1)

            # Insert patch into image
            image = insert_patch_subpixel(image, data[i], (IMG_WIDTH - PATCH_WIDTH / 2,
                                                           h * step_size_height + PATCH_HEIGHT / 2))
            i += 1

        for w in range(PATCHES - 1):
            # Insert patches from the bottom border, may overlap more
            overlap_last_height = (PATCH_HEIGHT + (PATCHES - 2) * step_size_height) - (IMG_HEIGHT - PATCH_HEIGHT)

            if overlap_last_height > 0:
                # Create array of overlap with mean values from overlapping patches
                mean_overlap_height = cv2.addWeighted(data[i][:overlap_last_height], 0.5,
                                                      data[i - PATCHES][PATCH_HEIGHT - overlap_last_height:], 0.5, 0)

                # Insert array of overlap into patch where it overlaps
                rest = data[i][overlap_last_height:]
                data[i] = np.append(mean_overlap_height, rest, axis=0)

            # Insert patch into image
            image = insert_patch_subpixel(image, data[i], (w * step_size_width + PATCH_WIDTH / 2,
                                                           IMG_HEIGHT - PATCH_HEIGHT / 2))
            i += 1

        # Insert patch in the bottom right corner, may overlap more
        overlap_last_width = (PATCH_WIDTH + (PATCHES - 2) * step_size_width) - (IMG_WIDTH - PATCH_WIDTH)

        if overlap_last_width > 0:
            # Create array of overlap along x axis with mean values form overlapping patches
            mean_overlap_width = cv2.addWeighted(data[i][:, [i for i in range(0, overlap_last_width)]], 0.5,
                                                 data[i - 1][:, [i for i in range(PATCH_WIDTH - overlap_last_width,
                                                                                  PATCH_WIDTH)]], 0.5, 0)

            # Insert array of overlap into patch
            rest = data[i][:, [i for i in range(overlap_last_width, PATCH_WIDTH)]]
            data[i] = np.append(mean_overlap_width, rest, axis=1)

        overlap_last_height = (PATCH_HEIGHT + (PATCHES - 2) * step_size_height) - (IMG_HEIGHT - PATCH_HEIGHT)

        if overlap_last_height > 0:
            # Create array of overlap along y axis with mean values from overlapping patches
            mean_overlap_height = cv2.addWeighted(data[i][:overlap_last_height], 0.5,
                                                  data[i - PATCHES][PATCH_HEIGHT - overlap_last_height:], 0.5, 0)

            # Insert array of overlap into patch where it overlaps
            rest = data[i][overlap_last_height:]
            data[i] = np.append(mean_overlap_height, rest, axis=0)

        image = insert_patch_subpixel(image, data[i], (IMG_WIDTH - PATCH_WIDTH / 2, IMG_HEIGHT - PATCH_HEIGHT / 2))
        i += 1
        whole_images.append(
            image)  # All corresponding patches are pinned inside the image, therefore this image is finished

    return whole_images


def insert_patch_subpixel(img, patch, p):
    """
    Insert patch into image at a given pixel
    :param img: original image
    :param patch: patch from image to insert
    :param p: tupel of pixel (width/height) as position of center of patch inside image
    """
    ths = patch.shape[0] / 2
    xpmin = p[0] - ths
    ypmin = p[1] - ths
    Ho = np.array([[1, 0, xpmin],
                   [0, 1, ypmin],
                   [0, 0, 1]], dtype=float)

    w = img.shape[0]
    h = img.shape[1]
    img2 = cv2.warpPerspective(patch, Ho, (h, w), dst=img,
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_TRANSPARENT)
    return img2


def make_ndarray(data: list, convert=False):
    """
    Last change made to the data to make it an ndarray
    :param convert: convert data to float
    :param data: list of images or masks
    :return: ndarray of images or masks converted to float32
    """
    data_height = data[0].shape[0]
    data_width = data[0].shape[1]
    if len(data[0].shape) == 3:
        data_channels = data[0].shape[2]
        nd_data = np.zeros((len(data), data_height, data_width, data_channels), dtype=np.float32)

    else:
        nd_data = np.zeros((len(data), data_height, data_width), dtype=np.float32)

    if convert:
        for _ in range(len(data)):
            nd_data[_] = tf.keras.layers.Lambda(lambda x: x / 255)(data[_])

    else:
        for _ in range(len(data)):
            nd_data[_] = data[_]

    return nd_data


def filter_small_vessels(predictions, size=3):
    """
    Remove small vessels from prediction to keep only blood vessels with minimum thickness of 2
    :param predictions: predictions to be filtered
    :param size: minimum pixel size a vessel must be to not be filtered
    :return: filtered predictions as numpy array
    """
    filtered_preds = []

    for _ in range(predictions.shape[0]):
        filtered = ndimage.binary_opening(predictions[_], structure=np.ones((size, size)))
        filtered_preds.append(filtered)

    return make_ndarray(filtered_preds)


def load_model(model_path):
    return keras.models.load_model(model_path, custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})


def predict(used_model, data):
    predictions = used_model.predict(data)
    predictions = predictions[:, :, :, 0]
    predictions = recreate_from_patches(predictions)
    predictions = make_ndarray(predictions)

    return predictions


if __name__ == '__main__':
    # Set IMG_HEIGHT and IMG_WIDTH to max values in images, ideally all images are the same size
    IMG_HEIGHT = 270
    IMG_WIDTH = 320
    IMG_CHANNELS = 3

    # PATCHES are number of patches along x and y axis, therefore PATCHESxPATCHES different sub images
    PATCHES = 17

    # PATCH_HEIGHT and PATCH_WIDTH must be set according to the patch size of the pretrained model
    PATCH_HEIGHT = 64
    PATCH_WIDTH = 64

    # THRESHOLD is ideally set according to the pretrained model
    THRESHOLD = 0.41

    model = load_model('/path/to/Model/R2UNet-3Blocks-Size64-17x17')
    images = load_images('/path/to/Images/', IMG_HEIGHT, IMG_WIDTH, convert=True)
    masks = predict(model, images)

    # threshold the values of the masks
    for i in range(len(masks)):
        masks[i] = (masks[i] > THRESHOLD).astype(np.int)
