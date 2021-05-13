import tensorflow as tf
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


def gaussian_highpass(image):
    """
    Gaussian highpass filter with sigma=2 to reduce noise, returns black/white image
    :param image: image to predict on
    :return filtered image of same shape
    """
    lowpass = ndimage.gaussian_filter(image, 2)
    highpass = image - lowpass
    return highpass


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


def load_Data(img_path, mask_path):
    """
    Put all images and masks as they are into a list to work with
    :param img_path: path to the image files
    :param mask_path: path to the mask files
    :return: two lists containing all the images and masks respectively
    """
    image_files = glob(os.path.join(img_path, '*.*'))
    mask_files = glob(os.path.join(mask_path, '*.*'))
    image_files.sort()
    mask_files.sort()
    images_list = []
    masks_list = []

    for _ in range(len(image_files)):
        image = cv2.imread(image_files[_])
        mask = cv2.imread(mask_files[_])
        images_list.append(image)
        masks_list.append(mask)

    return images_list, masks_list


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


def data_augmentation(image_data, mask_data, rotate=False, vertical_flip=False, horizontal_flip=False):
    """
    Perform data augmentation on given images and masks
    :param image_data: images to augment
    :param mask_data: masks to augment the same way as the corresponding image
    :param rotate: set True if rotation is wanted
    :param vertical_flip: set True if vertical flip is wanted
    :param horizontal_flip: set True if horizontal flip is wanted
    :return: two ndarrays holding the images and the masks
    """
    aug_images = []
    aug_masks = []

    for _ in range(len(image_data)):
        if rotate:
            rotation = A.RandomRotate90(p=1)
            rotated_data = rotation(image=image_data[_], mask=mask_data[_])
            rotated_image = rotated_data['image']
            rotated_mask = rotated_data['mask']
            aug_images.append(rotated_image)
            aug_masks.append(rotated_mask)

        if vertical_flip:
            flip_v = A.VerticalFlip(p=1)
            vertical_data = flip_v(image=image_data[_], mask=mask_data[_])
            vertical_image = vertical_data['image']
            vertical_mask = vertical_data['mask']
            aug_images.append(vertical_image)
            aug_masks.append(vertical_mask)

        if horizontal_flip:
            flip_h = A.HorizontalFlip(p=1)
            horizontal_data = flip_h(image=image_data[_], mask=mask_data[_])
            horizontal_image = horizontal_data['image']
            horizontal_mask = horizontal_data['mask']
            aug_images.append(horizontal_image)
            aug_masks.append(horizontal_mask)

    nd_images = make_ndarray(aug_images)
    nd_masks = make_ndarray(aug_masks)
    #nd_images = np.zeros((len(aug_images), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    #nd_masks = np.zeros((len(aug_masks), IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

    #for _ in range(len(aug_images)):  # Load into ndarray
    #    nd_images[_] = aug_images[_]
    #    nd_masks[_] = aug_masks[_]  # load mask without channel variable

    return nd_images, nd_masks


def roi_data(image_data, mask_data, roi_data_path):
    """ Put Region Of Interest masks over predictions and the corresponding mask """
    # Get ROI masks from files
    roi_files = glob(roi_data_path + '*.png')
    roi_files.sort()
    roi_masks = []
    for file in roi_files:
        roi = cv2.imread(file)
        roi_masks.append(roi)

    for _ in range(len(roi_masks)):
        roi_masks[_] = make_border(roi_masks[_], image_data.shape[1], image_data.shape[2])

    roi_masks = make_ndarray(roi_masks)[:, :, :, 0]

    for _ in range(len(image_data)):
        image_data[_] *= roi_masks[_]
        mask_data[_] *= roi_masks[_]

    return image_data, mask_data


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


IMG_WIDTH = 320
IMG_HEIGHT = 270
IMG_CHANNELS = 3

PATCHES = 17  # Number of patches along x and y axis, original image sliced into PATCHESxPATCHES different patches
PATCH_HEIGHT = 64
PATCH_WIDTH = 64

DATA_AUGMENTATION = 3  # Number of images generated on test images using data augmentation

BASE_IMG_PATH = os.path.join('..', '/path/to/working_directory/')

# Load Data
images_orig, masks_orig = load_Data(img_path=os.path.join(BASE_IMG_PATH, 'Images/'),
                                    mask_path=os.path.join(BASE_IMG_PATH, 'Masks/'))

# Add borders to make all images and masks same size
for _ in range(len(images_orig)):
    images_orig[_] = make_border(images_orig[_], IMG_HEIGHT, IMG_WIDTH)

for _ in range(len(masks_orig)):
    masks_orig[_] = make_border(masks_orig[_], IMG_HEIGHT, IMG_WIDTH)

# Bilateral Filter
# for _ in range(len(images_orig)):
#    images_orig[_] = cv2.bilateralFilter(images_orig[_], 9, 75, 75)

# Make ndarrays to put into the model
images = make_ndarray(images_orig, convert=True)
masks = make_ndarray(masks_orig)[:, :, :, 0]

X_train, X_test, Y_train, Y_test = train_test_split(images, masks, test_size=0.1)

X_train = slice_patches(X_train, PATCH_HEIGHT, PATCH_WIDTH)
X_test = slice_patches(X_test, PATCH_HEIGHT, PATCH_WIDTH)
Y_train = slice_patches(Y_train, PATCH_HEIGHT, PATCH_WIDTH)
Y_test = slice_patches(Y_test, PATCH_HEIGHT, PATCH_WIDTH)

# Data Augmentation
aug_X_train, aug_Y_train = data_augmentation(X_train, Y_train, rotate=True, vertical_flip=True, horizontal_flip=True)

X_train = aug_X_train
Y_train = aug_Y_train

# load some difficult images as fixed test set to compare performances
testImages_orig, testMasks_orig = load_Data(os.path.join(BASE_IMG_PATH, 'TestData/TestImages/'),
                                            os.path.join(BASE_IMG_PATH, 'TestData/TestMasks/'))

# Make boarder around images and masks
for _ in range(len(testImages_orig)):
    testImages_orig[_] = make_border(testImages_orig[_], IMG_HEIGHT, IMG_WIDTH)

for _ in range(len(testMasks_orig)):
    testMasks_orig[_] = make_border(testMasks_orig[_], IMG_HEIGHT, IMG_WIDTH)

# Bilateral Filter Images
# for _ in range(len(testImages_orig)):
#    testImages_orig[_] = cv2.bilateralFilter(testImages_orig[_], 9, 75, 75)

# Put images and masks into ndarray to feed into model
testImages = make_ndarray(testImages_orig, convert=True)
testMasks = make_ndarray(testMasks_orig)[:, :, :, 0]

testImage_slices = slice_patches(testImages, PATCH_HEIGHT, PATCH_WIDTH)

testImages = testImage_slices


