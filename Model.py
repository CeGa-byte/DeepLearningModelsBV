from ModelData import *
from Plots import *


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


if __name__ == '__main__':
    # Build the UNet
    inputs = tf.keras.layers.Input((PATCH_HEIGHT, PATCH_WIDTH, IMG_CHANNELS))
    # s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)  # Convert values to floating point

    """
    Contraction part. 16 3x3 filters on input layer, Dropout between Convolution Layers, 2x2 Max Pooling,
    padding on edges == same.
    """
    c1 = tf.keras.layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Activation('relu')(c1)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(c1)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Activation('relu')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    p1 = tf.keras.layers.BatchNormalization()(p1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.Activation('relu')(c2)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c2)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.Activation('relu')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    p2 = tf.keras.layers.BatchNormalization()(p2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Activation('relu')(c3)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(c3)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Activation('relu')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    p3 = tf.keras.layers.BatchNormalization()(p3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.Activation('relu')(c4)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(c4)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.Activation('relu')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = tf.keras.layers.BatchNormalization()(p4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Activation('relu')(c5)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(c5)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Activation('relu')(c5)

    """
    Expansion part. Other way around of Contraction part. Concatenating layer from contraction and expansion 
    before convolution.
    """
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.BatchNormalization()(u6)
    u6 = tf.keras.layers.concatenate([u6, c4], axis=3)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.BatchNormalization()(c6)
    c6 = tf.keras.layers.Activation('relu')(c6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(c6)
    c6 = tf.keras.layers.BatchNormalization()(c6)
    c6 = tf.keras.layers.Activation('relu')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.BatchNormalization()(u7)
    u7 = tf.keras.layers.concatenate([u7, c3], axis=3)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.BatchNormalization()(c7)
    c7 = tf.keras.layers.Activation('relu')(c7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(c7)
    c7 = tf.keras.layers.BatchNormalization()(c7)
    c7 = tf.keras.layers.Activation('relu')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.BatchNormalization()(u8)
    u8 = tf.keras.layers.concatenate([u8, c2], axis=3)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.BatchNormalization()(c8)
    c8 = tf.keras.layers.Activation('relu')(c8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c8)
    c8 = tf.keras.layers.BatchNormalization()(c8)
    c8 = tf.keras.layers.Activation('relu')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.BatchNormalization()(u9)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.BatchNormalization()(c9)
    c9 = tf.keras.layers.Activation('relu')(c9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(c9)
    c9 = tf.keras.layers.BatchNormalization()(c9)
    c9 = tf.keras.layers.Activation('relu')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model_unet = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model_unet.compile(optimizer='adam', loss=dice_coef_loss, metrics=['accuracy'])
    model_unet.summary()

    # Model checkpoint for unexpected crash
    checkpointer = tf.keras.callbacks.ModelCheckpoint('Model_For_BloodVessels.h5', verbose=1, save_best_only=True)

    callbacks = [tf.keras.callbacks.EarlyStopping(patience=5,
                                                  monitor='val_loss')]  # Early stopping if validation loss does not decrease 3 consecutive epochs

    results = model_unet.fit(X_train, Y_train, batch_size=16, epochs=50, callbacks=callbacks, validation_data=(X_test, Y_test))

    plot_history(results, save=True, path=BASE_IMG_PATH + 'Model_Plots/Thesis/UNet/History.png')

    # Run model on test data set
    y_predict = model_unet.predict(testImages)
    y_predict = y_predict[:, :, :, 0]  # Get rid of last dimension to get 3 dimensional predictions
    y_predict = recreate_from_patches(y_predict)
    y_predict = make_ndarray(y_predict)
    testImages = recreate_from_patches(testImages)
    testImages = make_ndarray(testImages)

    # Plot figure with test images, test masks and the predictions
    plot_orig_mask_pred(testImages, testMasks, y_predict, save=True, path=BASE_IMG_PATH + 'Model_Plots/UNet/Predictions/')

    # Plot ROC Curve
    plot_ROC(testMasks, y_predict, save=True, path=BASE_IMG_PATH + 'Model_Plots/UNet/')

    # Get perfect threshold
    threshold = find_threshold(y_predict, testMasks)

    # Plot mean IoU values for different thresholds
    plot_thresholds(y_predict, testMasks, save=True, path=BASE_IMG_PATH + 'Model_Plots/UNet/')

    # Threshold predictions
    for i in range(len(y_predict)):
        y_predict[i] = (y_predict[i] > threshold).astype(np.int)

    # Overlap predictions and mask to showcase differences with different colors
    overlaps = overlap_pred_mask(y_predict, testMasks)

    # Plot image, mask over image, prediction over image and overlay mask/prediction in one figure
    plot_overlays(testImages_orig, testMasks, y_predict, threshold=threshold, save=True, path=BASE_IMG_PATH + 'Model_Plots/UNet/Overlays/')

