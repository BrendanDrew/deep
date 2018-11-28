import keras
import keras.models
import keras.layers
import keras.utils
import keras.callbacks
import keras.regularizers
import keras.layers.advanced_activations
import keras.utils.np_utils
import keras.preprocessing.image
import matplotlib.pyplot as plt
import random
import tarfile
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np
import numpy.random as npr
import math
import sys
import gc

if __name__ == '__main__':
    main()

def extract_samples(filename, patch_size, samples_per_image, sample_probability=0.05):
    samples = []

    prng = random.Random()

    print('Examining [{}]'.format(filename))
    with tarfile.open(filename, 'r:*') as tar:
        for thing in tar:
            #if thing.name <= 'data_large/c/':
                if thing.isreg() and prng.uniform(0, 1) <= sample_probability:
                    with tar.extractfile(thing) as jpeg:
                        print('Processing [{} :: {}]'.format(filename, thing.name))
                        with Image.open(jpeg) as image:
                            pix = np.array(image)

                            if len(pix.shape) == 2:
                                print('[{} :: {}] appears to be grayscale -- skipping'.format(filename, thing.name))
                                continue

                            if pix.shape[0] > patch_size[0] and pix.shape[1] > patch_size[1]:
                                for s in range(samples_per_image):
                                    offset_row = math.floor(prng.uniform(0, pix.shape[0] - patch_size[0]))
                                    offset_col = math.floor(prng.uniform(0, pix.shape[1] - patch_size[1]))
                                    chip = pix[offset_row:(offset_row + patch_size[0]), offset_col:(offset_col + patch_size[1]), :].copy()
                                    
                                    if (patch_size[0], patch_size[1], 3) == chip.shape:
                                        samples.append(chip)
                                    else:
                                        print('Warning: offset=(row={}, col={}), patch size=(rows={}, cols={}), image shape=({}), chip shape=({})'.format(offset_row, offset_col, patch_size[0], patch_size[1], pix.shape, chip.shape))
            #else:
            #    break
                                    

    return samples


def network_architecture(regularization=0, patch_size=(128, 128)):
    convolution_layer_common_arguments = {
        'data_format': 'channels_last',
        'use_bias': True,
    }

    batch_common_arguments = {
        'axis': 3
    }

    dropout_common_arguments = {
        'data_format': 'channels_last',
        'rate': 0.25
    }

    num_representation_layers = 4
    num_reconstruction_layers = num_representation_layers + 2
    
    if regularization > 0:
        convolution_layer_common_arguments['activity_regularizer'] = keras.regularizers.l1(regularization)

    m = keras.Sequential()
    m.add(keras.layers.BatchNormalization(input_shape=(patch_size[0], patch_size[1], 3), **batch_common_arguments))
    m.add(keras.layers.Conv2D(filters=8, kernel_size=(7, 7), **convolution_layer_common_arguments))
    m.add(keras.layers.BatchNormalization(**batch_common_arguments))
    m.add(keras.layers.ReLU())
    m.add(keras.layers.SpatialDropout2D(**dropout_common_arguments))

    for i in range(num_representation_layers):
        m.add(keras.layers.Conv2D(filters=32 * (i * 2 + 1), kernel_size=(3, 3), strides=(2, 2), **convolution_layer_common_arguments))
        m.add(keras.layers.BatchNormalization(**batch_common_arguments))
        m.add(keras.layers.ReLU())
        m.add(keras.layers.SpatialDropout2D(**dropout_common_arguments))

    print('Size before final representation: {}'.format(m.layers[-1].output_shape))

    m.add(keras.layers.Conv2D(filters=1024, kernel_size=(6, 6), **convolution_layer_common_arguments))
    m.add(keras.layers.BatchNormalization(**batch_common_arguments))
    m.add(keras.layers.ReLU(name='encoder_output'))
    m.add(keras.layers.SpatialDropout2D(**dropout_common_arguments))
    

    print('Size after global max pooling: {}'.format(m.layers[-1].output_shape))

    upsampling_common_arguments = {
        'data_format': 'channels_last',
        'size': (2, 2)
    }

    for i in range(num_reconstruction_layers):
        m.add(keras.layers.UpSampling2D(**upsampling_common_arguments))
        m.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', **convolution_layer_common_arguments))
        m.add(keras.layers.BatchNormalization(**batch_common_arguments))
        m.add(keras.layers.ReLU())
        m.add(keras.layers.SpatialDropout2D(**dropout_common_arguments))
        print('Decoder stage [{}] output size: [{}]'.format(i, m.layers[-1].output_shape))

    m.add(keras.layers.UpSampling2D(**upsampling_common_arguments))
    m.add(keras.layers.Conv2D(filters=3, kernel_size=(5, 5), padding='same', **convolution_layer_common_arguments))
    m.add(keras.layers.BatchNormalization(**batch_common_arguments))
    m.add(keras.layers.ReLU())
    print('Decoder final stage size: {}'.format(m.layers[-1].output_shape))

    m.compile(loss='logcosh',
              optimizer='nadam',
              metrics=['mae', 'mean_squared_error'])

    keras.utils.plot_model(m, to_file='autoencoder-model.pdf', show_shapes=True, show_layer_names=True)
    
    return m



def main():
    patch_size = (128, 128)

    m = network_architecture(patch_size=patch_size)

    m.summary()
    
    samples = np.stack(extract_samples(sys.argv[1], patch_size, samples_per_image=1, sample_probability=0.05), axis=0)

    train_percentage = 0.7
    validation_percentage = 0.15
    assignments = npr.ranf((samples.shape[0],))

    print('Splitting data into train/test/validate')
    train_data = samples[assignments < train_percentage, :, :, :].copy()
    val_data = samples[np.logical_and(train_percentage <= assignments, assignments < train_percentage + validation_percentage), :, : , :].copy()
    test_data = samples[assignments >= train_percentage + validation_percentage, :, :, :].copy()
    del samples
    gc.collect()

    print('Train: [{}], validate: [{}], test: [{}]'.format(train_data.shape, val_data.shape, test_data.shape))

    epochs = 2000
    batch_size = 256
    
    history = m.fit(
        x=train_data,
        y=train_data,
        batch_size=batch_size,
        validation_data=(val_data, val_data),
        epochs=epochs,
        shuffle=True,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_mean_squared_error',
                                          verbose=1,
                                          mode='min',
                                          patience=150,
                                          restore_best_weights=True),
        ])


    with PdfPages('autoencoder-analysis.pdf') as pdf:
        f = plt.figure(figsize=(11, 8.5), dpi=600)
        a = f.gca()
        a.set_title('Loss')
        a.plot(history.history['loss'], label='Training')
        a.plot(history.history['val_loss'], label='Validation')
        a.set_ylabel('Loss')
        a.set_xlabel('Epoch')
        a.legend()
        pdf.savefig(f)
        plt.close(f)

        f = plt.figure(figsize=(11, 8.5), dpi=600)
        a = f.gca()
        a.set_title('Mean Squared Error')
        a.plot(history.history['mean_squared_error'], label='Training')
        a.plot(history.history['val_mean_squared_error'], label='Validation')
        a.set_ylabel('MAE')
        a.set_xlabel('Epoch')
        a.legend()
        pdf.savefig(f)
        plt.close(f)

        f = plt.figure(figsize=(11, 8.5), dpi=600)
        a = f.gca()
        a.set_title('Mean Absolute Error')
        a.plot(history.history['mean_absolute_error'], label='Training')
        a.plot(history.history['val_mean_absolute_error'], label='Validation')
        a.set_ylabel('MSE')
        a.set_xlabel('Epoch')
        a.legend()
        pdf.savefig(f)
        plt.close(f)
