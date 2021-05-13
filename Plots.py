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
from sklearn.metrics import roc_auc_score, roc_curve, auc


def calc_iou(y_true, y_pred):
    """ Calculate Intersection Over Union for binarized prediction and mask """
    return np.sum(np.logical_and(y_true, y_pred)) / np.sum(np.logical_or(y_true, y_pred))


def plot_orig_predoverlay(images, predictions, save=False, path=None):
    """ Plot original image and prediction next to each other """
    for _ in range(len(images)):
        overlay = create_overlay(images[_] * 255, predictions[_], color=[0, 255, 255])
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        ax[0].set_title('Original Image')
        ax[1].set_title('Predicted Mask')

        ax[0].imshow(images[_], cmap='gray')
        ax[1].imshow(overlay)

        if save:
            plt.savefig(path + 'Image' + str(_) + '.png')
        plt.show()

    return None


def create_overlay(image1, image2, color):
    """
    Lay image2 over image1
    :param image1: given image as ndarray
    :param image2: image to lay over
    :param color: array with 3 inputs range 0-255
    :return: colored image where image2 overlays image1
    """
    overlay_image = np.zeros(image2.shape + (3,), dtype=np.int)
    for row in range(len(image1)):
        for col in range(len(image1[0])):
            if image2[row][col] == 1:
                overlay_image[row][col] = color
            else:
                background = image1[row][col]
                overlay_image[row][col] = list(background)

    return overlay_image


def overlap_pred_mask(predictions, groundTruth_masks):
    overlap_images = np.zeros(predictions.shape + (3,), dtype=np.int)  # Shape of the overlap ndarray is same as predictions but 3 channel images
    for _ in range(len(predictions)):
        img = predictions[_]
        mask = groundTruth_masks[_]
        for row in range(len(img)):
            for col in range(len(img[0])):
                if img[row][col] == 1 and mask[row][col] == 1:  # Prediction is correct
                    overlap_images[_][row][col] = [255, 255, 0]  # Yellow
                elif img[row][col] == 0 and mask[row][col] == 1:  # False negative
                    overlap_images[_][row][col] = [0, 255, 255]  # Blue
                elif img[row][col] == 1 and mask[row][col] == 0:  # False positive
                    overlap_images[_][row][col] = [255, 0, 255]  # Pink

    return overlap_images


def plot_history(results, save=False, path=None):
    """
    Plot the history of the model results
    :param results: results of the fitted model
    :param save: bool to specify if plot should be saved
    :param path: path to save the figure if wanted
    :return: None
    """
    fig = plt.figure(figsize=(10, 3))
    # Plot training & validation accuracy values
    plt.subplot(121)
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    if save:
        plt.savefig(path)
    plt.show()

    return None


def plot_orig_mask_pred(images, groundTruth_masks, predictions, save=False, path=None):
    """
    plot original image, mask and model prediction
    :param images: original image
    :param groundTruth_masks: mask
    :param predictions: prediction from model
    :param save: bool to specify if plot should be saved
    :param path: path to save the plot
    :return: None
    """
    for i in range(images.shape[0]):
        # Print test images, test masks and model predictions
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].set_title('Original Image')
        ax[1].set_title('Mask')
        ax[2].set_title('Predicted Mask')

        ax[0].imshow(images[i], cmap='gray')
        ax[1].imshow(groundTruth_masks[i], cmap='gray')
        ax[2].imshow(predictions[i], cmap='gray')

        if save:
            plt.savefig(path + 'Image' + str(i) + '.png')
        plt.show()

    return None


def plot_pred_mask_overlay(predictions, groundTruth_masks, threshold, save=False, path=None):
    """ Plot prediction, mask and overlay prediction/mask """
    overlaps = overlap_pred_mask(predictions=predictions, groundTruth_masks=groundTruth_masks)
    for _ in range(len(predictions)):
        iou = calc_iou(groundTruth_masks[_], predictions[_])
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].set_title('Prediction (Threshold = %s)' % round(threshold, 2))
        ax[1].set_title('Mask')
        ax[2].set_title('Overlap Image (IoU = %s)' % round(iou, 2))

        ax[0].imshow(predictions[_], cmap='gray')
        ax[1].imshow(groundTruth_masks[_], cmap='gray')
        ax[2].imshow(overlaps[_])

        if save:
            plt.savefig(path + 'PredMaskOverlay' + str(_) + '.png')
        plt.show()

    return None


def plot_thresholds(predictions, groundTruth_masks, save=False, path=None):
    """
    Plot mean Intersection over Union for different threshold
    :param predictions: predictions from model
    :param groundTruth_masks: masks
    :param save: bool to specify if plot should be saved
    :param path: path to save to
    :return: None
    """
    thresh_graph = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Different thresholds
    iou_graph = []
    for thresh in thresh_graph:
        ious = []
        for i in range(len(predictions)):
            img = predictions[i]
            pred = (img > thresh).astype(np.int)
            ious.append(calc_iou(groundTruth_masks[i], pred))
        mean_iou = statistics.mean(ious)
        iou_graph.append(mean_iou)

    # Plot threshold graph
    plt.figure()
    plt.title('Threshold Graph')
    plt.xlabel('Thresholds')
    plt.ylabel('Mean IoU over all Predictions')
    plt.plot(thresh_graph, iou_graph, color='red')

    if save:
        plt.savefig(path + 'ThresholdGraph.png')
    plt.show()

    return None


def plot_overlays(images, groundTruth_masks, predictions, threshold, save=False, path=None):
    """ Plot original image, mask over image, prediction over image and overlap prediction and mask """
    overlaps_preds_masks = overlap_pred_mask(predictions, groundTruth_masks)
    for _ in range(len(images)):
        overlay_image_mask = create_overlay(images[_], groundTruth_masks[_], color=[0, 255, 255])
        overlay_image_pred = create_overlay(images[_], predictions[_], color=[255, 0, 255])

        iou = calc_iou(groundTruth_masks[_], predictions[_])
        fig, ax = plt.subplots(1, 4, figsize=(18, 6))
        ax[0].set_title('Original Image')
        ax[1].set_title('Mask')
        ax[2].set_title('Predicted Mask (Threshold = %s)' % round(threshold, 2))
        ax[3].set_title('Overlap Label/Prediction (IoU = %s)' % round(iou, 2))

        ax[0].imshow(images[_], cmap='gray')
        ax[1].imshow(overlay_image_mask, cmap='gray')
        ax[2].imshow(overlay_image_pred, cmap='gray')
        ax[3].imshow(overlaps_preds_masks[_])

        if save:
            plt.savefig(path + 'Overlays' + str(_) + '.png')

        plt.show()

    return None


def plot_ROC(groundTruth, predictions, save=False, path=None):
    """ Plot ROC curve and Area under Curve """
    false_positive, true_positive, threshold = roc_curve(groundTruth.ravel(), predictions.ravel())
    area_under_curve = auc(false_positive, true_positive)

    plt.plot(false_positive, true_positive, label='ROC (AUC = %.4f)' % area_under_curve)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.legend()

    if save:
        plt.savefig(path + 'ROC_Curve')

    plt.show()

    return None


def plot_filtered(predictions, filtered, save=False, path=None):
    """ Plot predictions and filtered for small vessels predictions """
    for _ in range(predictions.shape[0]):
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        ax[0].set_title('Prediction')
        ax[1].set_title('Filtered')

        ax[0].imshow(predictions[_], cmap='gray')
        ax[1].imshow(filtered[_], cmap='gray')

        if save:
            plt.savefig(path + 'Filtered' + str(_) + '.png')

        plt.show()
    return None
