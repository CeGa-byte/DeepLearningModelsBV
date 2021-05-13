from Model import calc_iou, IoU, jaccard_distance, dice_coef, dice_coef_loss, find_threshold
from Plots import *
from ModelData import *
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate, Input, add
from tensorflow.keras.models import Model


def batchnorm_relu(inputs):
    """ Batch Normalization & ReLU """
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    return x


def recurrent_residual_block(inputs, num_filters, strides=1, dilation=1):
    """ Skip layer """
    skip_layer = Conv2D(num_filters, 1, padding="same", strides=strides)(inputs)
    layer = skip_layer

    """ Convolutional Layers """
    x = Conv2D(num_filters, 3, padding="same", strides=strides, dilation_rate=dilation)(layer)
    x = batchnorm_relu(x)
    x = Conv2D(num_filters, 3, padding="same", strides=strides, dilation_rate=dilation)(add([x, layer]))
    x = batchnorm_relu(x)
    x = Conv2D(num_filters, 3, padding="same", strides=strides, dilation_rate=dilation)(add([x, layer]))
    x = batchnorm_relu(x)

    layer = x
    x = Conv2D(num_filters, 3, padding="same", strides=strides, dilation_rate=dilation)(layer)
    x = batchnorm_relu(x)
    x = Conv2D(num_filters, 3, padding="same", strides=strides, dilation_rate=dilation)(add([x, layer]))
    x = batchnorm_relu(x)
    x = Conv2D(num_filters, 3, padding="same", strides=strides, dilation_rate=dilation)(add([x, layer]))
    x = batchnorm_relu(x)

    """ Addition """
    out_layer = add([x, skip_layer])
    return out_layer


def decoder_block(inputs, skip_features, num_filters):
    """ Decoder Block """
    x = UpSampling2D((2, 2))(inputs)
    x = Concatenate()([x, skip_features])
    x = recurrent_residual_block(x, num_filters, strides=1)
    return x


def dilation_block(inputs, num_filters):
    """
    Feature maps of previous layer are followed by convolution, batch normalization and ReLU and then passed
    separately to each layer in the dilation box containing different dilation rates from 2 to the power of 0 to 2 to
    the power of 5.
    """
    dilation1 = Conv2D(num_filters, 3, padding='same', dilation_rate=1)(inputs)
    dilation1 = batchnorm_relu(dilation1)

    dilation2 = Conv2D(num_filters, 3, padding='same', dilation_rate=2)(inputs)
    dilation2 = batchnorm_relu(dilation2)

    dilation4 = Conv2D(num_filters, 3, padding='same', dilation_rate=4)(inputs)
    dilation4 = batchnorm_relu(dilation4)

    dilation8 = Conv2D(num_filters, 3, padding='same', dilation_rate=8)(inputs)
    dilation8 = batchnorm_relu(dilation8)

    dilation16 = Conv2D(num_filters, 3, padding='same', dilation_rate=16)(inputs)
    dilation16 = batchnorm_relu(dilation16)

    dilation32 = Conv2D(num_filters, 3, padding='same', dilation_rate=32)(inputs)
    dilation32 = batchnorm_relu(dilation32)

    dilation_sum = add([dilation1, dilation2, dilation4, dilation8, dilation16, dilation32])

    return dilation_sum


def build_r2unet(input_shape):
    """ RESUNET Architecture """
    inputs = Input(input_shape)

    """ Encoder 1, 2, 3 ... with skip and pool layers """
    s1 = recurrent_residual_block(inputs, 16, strides=1)
    p1 = MaxPool2D(pool_size=(2, 2))(s1)
    s2 = recurrent_residual_block(p1, 32, strides=1)
    p2 = MaxPool2D(pool_size=(2, 2))(s2)
    s3 = recurrent_residual_block(p2, 64, strides=1)
    p3 = MaxPool2D(pool_size=(2, 2))(s3)
    # s4 = recurrent_residual_block(p3, 128, strides=1)
    # p4 = MaxPool2D(pool_size=(2, 2))(s4)

    """ Bridge """
    b = recurrent_residual_block(p3, 128, strides=1)
    # b = dilation_block(p3, 128)

    """ Decoder 1, 2, 3 ..."""
    # x = decoder_block(b, s4, 128)
    x = decoder_block(b, s3, 64)
    x = decoder_block(x, s2, 32)
    x = decoder_block(x, s1, 16)

    """ Classifier """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(x)

    """ Model """
    resunet = Model(inputs, outputs, name="RESUNET")

    return resunet


if __name__ == '__main__':
    # Build the R2UNet
    model_r2unet = build_r2unet((PATCH_HEIGHT, PATCH_WIDTH, IMG_CHANNELS))
    model_r2unet.summary()

    model_r2unet.compile(optimizer='adam', loss=dice_coef_loss, metrics=['accuracy'])

    # Model checkpoint for unexpected crash
    checkpointer = tf.keras.callbacks.ModelCheckpoint('R2UNetModel_For_BloodVessels.h5', verbose=1, save_best_only=True)

    callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss')]  # Early stopping if validation loss does not decrease 3 consecutive epochs

    results = model_r2unet.fit(X_train, Y_train, batch_size=16, epochs=50, callbacks=callbacks,
                               validation_data=(X_test, Y_test))

    plot_history(results, save=False, path=BASE_IMG_PATH + 'Model_Plots/History.png')

    y_predict = model_r2unet.predict(testImages)
    y_predict = y_predict[:, :, :, 0]  # Get rid of last dimension
    y_predict = recreate_from_patches(y_predict)
    y_predict = make_ndarray(y_predict)
    testImages = recreate_from_patches(testImages)
    testImages = make_ndarray(testImages)

    # Plot figure with test images, test masks and the predictions
    plot_orig_mask_pred(testImages, testMasks, y_predict, save=False,
                        path=BASE_IMG_PATH + 'Model_Plots/Thesis/R2UNet/Predictions/')

    # Plot ROC curve
    plot_ROC(testMasks, y_predict, save=False, path=BASE_IMG_PATH + 'Model_Plots/R2UNet/')

    # Get perfect threshold
    threshold = find_threshold(y_predict, testMasks)

    # Plot mean IoU values for different thresholds
    plot_thresholds(y_predict, testMasks, save=False, path=BASE_IMG_PATH + 'Model_Plots/R2UNet/')

    # Threshold predictions
    for i in range(len(y_predict)):
        y_predict[i] = (y_predict[i] > threshold).astype(np.int)

    # Overlap predictions and mask to showcase differences with different colors
    # overlaps = overlap_pred_mask(y_predict, testMasks)

    # Plot image, mask over image, prediction over image and overlay mask/prediction in one figure
    plot_overlays(testImages_orig, testMasks, y_predict, threshold=threshold, save=False,
                  path=BASE_IMG_PATH + 'Model_Plots/R2UNet/Overlays/')

    # Show predictions, masks and overlap images
    # plot_pred_mask_overlay(y_predict, testMasks, threshold=threshold, save=False, path=BASE_IMG_PATH + '')
