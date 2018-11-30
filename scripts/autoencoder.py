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
import math
import sys

if __name__ == '__main__':
    main()

def extract_samples(filename, patch_size, seed, sample_probability=0.05, batch_size=100):
    prng = random.Random()
    prng.seed(seed)

    batch = ([], [])
    with tarfile.open(filename, 'r:*') as tar:
        while True:
            for thing in tar:
                if thing.isreg() and prng.uniform(0, 1) <= sample_probability:
                    with tar.extractfile(thing) as jpeg:
                        with Image.open(jpeg) as image:
                            pix = np.array(image)

                            if 3 == len(pix.shape) and 3 == pix.shape[2] and pix.shape[0] > patch_size[0] and pix.shape[1] > patch_size[1]:
                                offset_row = math.floor(prng.uniform(0, pix.shape[0] - patch_size[0]))
                                offset_col = math.floor(prng.uniform(0, pix.shape[1] - patch_size[1]))
                                chip = pix[offset_row:(offset_row + patch_size[0]), offset_col:(offset_col + patch_size[1]), :].astype(np.float32) / 256

                                if (patch_size[0], patch_size[1], 3) == chip.shape:
                                    batch[0].append(chip)
                                    batch[1].append(chip)

                                    if len(batch[0]) == batch_size:
                                        yield np.stack(batch[0], axis=0), np.stack(batch[1], axis=0)
                                        batch[0].clear()
                                        batch[1].clear()


def compute_num_representation_layers(patch_size, representation_kernel_size, representation_stride):

    layers = 0

    while ((patch_size[0] - 2 * representation_kernel_size[0] // 2) // representation_stride[0] > representation_kernel_size[0]) and ((patch_size[1] - 2 * representation_kernel_size[1] // 2) // representation_stride[1] > representation_kernel_size[1]):
        layers += 1
        patch_size = (patch_size[0] - 2 * representation_kernel_size[0] // 2) // representation_stride[0], (patch_size[1] - 2 * representation_kernel_size[1] // 2) // representation_stride[1]

    return layers, patch_size


def network_architecture(representation_size=256, patch_size=(128, 128)):
    convolution_layer_common_arguments = {
        'data_format': 'channels_last',
        'use_bias': True,
    }

    batch_common_arguments = {
        'axis': 3
    }

    dropout_common_arguments = {
        'data_format': 'channels_last',
        'rate': 0.125
    }

    upsampling_common_arguments = {
        'data_format': 'channels_last',
        'size': (2, 2)
    }

    representation_kernel_size = (3, 3)
    representation_stride = (2, 2)

    num_representation_layers, representation_shape = compute_num_representation_layers(patch_size, representation_kernel_size, representation_stride)

    print('Creating network architecture with [{}] convolutional representation layers and final shape of [{}]'.format(num_representation_layers, representation_shape))

    m = keras.models.Sequential()
    m.add(keras.layers.BatchNormalization(**batch_common_arguments, input_shape=(*patch_size, 3)))

    for i in range(num_representation_layers):
        m.add(keras.layers.Conv2D(filters = (i + 1) * 32,
                                  kernel_size=representation_kernel_size,
                                  strides=representation_stride,
                                  **convolution_layer_common_arguments))
        m.add(keras.layers.ReLU())
        m.add(keras.layers.SpatialDropout2D(**dropout_common_arguments))
        m.add(keras.layers.BatchNormalization(**batch_common_arguments))

    try:
        m.add(keras.layers.Conv2D(filters = (num_representation_layers + 1) * 32,
                                  kernel_size=representation_kernel_size,
                                  **convolution_layer_common_arguments))
        m.add(keras.layers.ReLU())
        m.add(keras.layers.SpatialDropout2D(**dropout_common_arguments))
        m.add(keras.layers.BatchNormalization(**batch_common_arguments))
    except:
        pass

    m.add(keras.layers.Conv2D(filters = 3 * (num_representation_layers + 1) * 32,
                              kernel_size=(1, 1),
                              **convolution_layer_common_arguments))
    m.add(keras.layers.ReLU())
    m.add(keras.layers.SpatialDropout2D(0.1))
    m.add(keras.layers.BatchNormalization(**batch_common_arguments))

    # Glue everything together
    m.add(keras.layers.Flatten(data_format='channels_last'))

    # And force it all through a pretty severe bottleneck into a final dense layer
    m.add(keras.layers.Dense(representation_size, activation='sigmoid', use_bias='True', name='encoder_output'))

    m.add(keras.layers.Reshape((1, 1, representation_size)))

    print('Encoder output shape: [{}]'.format(m.layers[-1].output_shape))

    # Now we reconstruct (hopefully)
    for i in range(7):
        m.add(keras.layers.BatchNormalization(**batch_common_arguments))
        m.add(keras.layers.UpSampling2D(**upsampling_common_arguments))
        m.add(keras.layers.Conv2D(filters=32, kernel_size=representation_kernel_size, **convolution_layer_common_arguments, padding='same'))
        m.add(keras.layers.ReLU())

    # Explode available channels
    m.add(keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same', **convolution_layer_common_arguments))
    m.add(keras.layers.ReLU())
    m.add(keras.layers.BatchNormalization(**batch_common_arguments))

    # And, finally, we get back to the correct number of channels.
    m.add(keras.layers.Conv2D(filters=3, kernel_size=(5, 5), activation='sigmoid', data_format='channels_last', padding='same'))
    print('Reconstructed shape: [{}]'.format(m.layers[-1].output_shape))


    m.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['mae', 'mse'])


    keras.utils.plot_model(m, to_file='autoencoder-model.pdf', show_shapes=True, show_layer_names=True)

    return m



def main():
    patch_size = (128, 128)

    with PdfPages('autoencoder-analysis.pdf') as pdf:
        for representation_size in [2048, 1024, 512, 256, 128, 64]:
            try:
                m = network_architecture(representation_size=representation_size, patch_size=patch_size)

                epochs = 2000
                batch_size = 32

                history = m.fit_generator(
                    extract_samples(sys.argv[1], patch_size, 0xdeadbeef, 0.1, batch_size=batch_size),
                    steps_per_epoch=500,
                    validation_data=extract_samples(sys.argv[1], patch_size, 0xc0ffee11, batch_size=batch_size),
                    validation_steps=100,
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

                f = plt.figure(figsize=(11, 8.5), dpi=600)
                a = f.gca()
                a.set_title('Loss (size={})'.format(representation_size))
                a.plot(history.history['loss'], label='Training')
                a.plot(history.history['val_loss'], label='Validation')
                a.set_ylabel('Loss')
                a.set_xlabel('Epoch')
                a.legend()
                pdf.savefig(f)
                plt.close(f)

                f = plt.figure(figsize=(11, 8.5), dpi=600)
                a = f.gca()
                a.set_title('Mean Squared Error (size={})'.format(representation_size))
                a.plot(history.history['mean_squared_error'], label='Training')
                a.plot(history.history['val_mean_squared_error'], label='Validation')
                a.set_ylabel('MAE')
                a.set_xlabel('Epoch')
                a.legend()
                pdf.savefig(f)
                plt.close(f)

                f = plt.figure(figsize=(11, 8.5), dpi=600)
                a = f.gca()
                a.set_title('Mean Absolute Error (size={})'.format(representation_size))
                a.plot(history.history['mean_absolute_error'], label='Training')
                a.plot(history.history['val_mean_absolute_error'], label='Validation')
                a.set_ylabel('MSE')
                a.set_xlabel('Epoch')
                a.legend()
                pdf.savefig(f)
                plt.close(f)
            except:
                print('Failed to build / train / etc. model for size {}; continuing'.format(representation_size))
