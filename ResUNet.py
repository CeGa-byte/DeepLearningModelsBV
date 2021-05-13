from Model import calc_iou, IoU, jaccard_distance, dice_coef, dice_coef_loss, find_threshold
from Plots import *
from ModelData import *
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate, Input
from tensorflow.keras.models import Model


def batchnorm_relu(inputs):
    """ Batch Normalization & ReLU """
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    return x


def residual_block(inputs, num_filters, strides=1):
    """ Convolutional Layers """
    x = batchnorm_relu(inputs)
    x = Conv2D(num_filters, 3, padding="same", strides=strides)(x)
    x = batchnorm_relu(x)
    x = Conv2D(num_filters, 3, padding="same", strides=1)(x)

    """ Shortcut Connection (Identity Mapping) """
    s = Conv2D(num_filters, 1, padding="same", strides=strides)(inputs)

    """ Addition """
    x = x + s
    return x


def decoder_block(inputs, skip_features, num_filters):
    """ Decoder Block """
    x = UpSampling2D((2, 2))(inputs)
    x = Concatenate()([x, skip_features])
    x = residual_block(x, num_filters, strides=1)
    return x


def build_resunet(input_shape):
    """ RESUNET Architecture """
    inputs = Input(input_shape)

    """ Endoder 1 """
    x = Conv2D(16, 3, padding="same", strides=1)(inputs)
    x = batchnorm_relu(x)
    x = Conv2D(16, 3, padding="same", strides=1)(x)
    s = Conv2D(16, 1, padding="same")(inputs)
    s1 = x + s
    p1 = MaxPool2D(pool_size=(2, 2))(s1)

    """ Encoder 2, 3 ..."""
    s2 = residual_block(p1, 32, strides=1)
    p2 = MaxPool2D(pool_size=(2, 2))(s2)
    s3 = residual_block(p2, 64, strides=1)
    p3 = MaxPool2D(pool_size=(2, 2))(s3)
    s4 = residual_block(p3, 128, strides=1)
    p4 = MaxPool2D(pool_size=(2, 2))(s4)
    s5 = residual_block(p4, 256, strides=1)
    p5 = MaxPool2D(pool_size=(2, 2))(s5)

    """ Bridge """
    b = residual_block(p5, 512, strides=1)

    """ Decoder 1, 2, 3 ... """
    x = decoder_block(b, s5, 256)
    x = decoder_block(x, s4, 128)
    x = decoder_block(x, s3, 64)
    x = decoder_block(x, s2, 32)
    x = decoder_block(x, s1, 16)

    """ Classifier """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(x)

    """ Model """
    model = Model(inputs, outputs, name="RESUNET")

    return model


if __name__ == '__main__':
    # Build the ResUNet
    model_resunet = build_resunet((PATCH_HEIGHT, PATCH_WIDTH, IMG_CHANNELS))
    model_resunet.summary()

    model_resunet.compile(optimizer='adam', loss=dice_coef_loss, metrics=['accuracy', tf.keras.metrics.MeanIoU(2)])

    # Model checkpoint for unexpected crash
    checkpointer = tf.keras.callbacks.ModelCheckpoint('ResUNetModel_For_BloodVessels.h5', verbose=1, save_best_only=True)

    callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss')]  # Early stopping if validation loss does not decrease 3 consecutive epochs

    results = model_resunet.fit(X_train, Y_train, batch_size=16, epochs=50, callbacks=callbacks,
                                validation_data=(X_test, Y_test))

    plot_history(results, save=False, path=BASE_IMG_PATH + 'Model_Plots/ResUNet/History.png')

    y_predict = model_resunet.predict(testImages)
    y_predict = y_predict[:, :, :, 0]  # Get rid of last dimension to get 3 dimensional predictions
    y_predict = recreate_from_patches(y_predict)
    y_predict = make_ndarray(y_predict)
    testImages = recreate_from_patches(testImages)
    testImages = make_ndarray(testImages)

    # Plot figure with test images, test masks and the predictions
    plot_orig_mask_pred(testImages, testMasks, y_predict, save=False,
                        path=BASE_IMG_PATH + 'Model_Plots/ResUNet/Predictions/')

    # Plot ROC Curve
    plot_ROC(testMasks, y_predict, save=False, path=BASE_IMG_PATH + 'Model_Plots/ResUNet/')

    # Get perfect threshold
    threshold = find_threshold(y_predict, testMasks)

    # Plot mean IoU values for different thresholds
    plot_thresholds(y_predict, testMasks, save=False, path=BASE_IMG_PATH + 'Model_Plots/ResUNet/')

    # Threshold predictions
    for i in range(len(y_predict)):
        y_predict[i] = (y_predict[i] > threshold).astype(np.int)

    # Overlap predictions and mask to showcase differences with different colors
    overlaps = overlap_pred_mask(y_predict, testMasks)

    # Plot image, mask over image, prediction over image and overlay mask/prediction in one figure
    plot_overlays(testImages_orig, testMasks, y_predict, threshold=threshold, save=False,
                  path=BASE_IMG_PATH + 'Model_Plots/ResUNet/Overlays/')
