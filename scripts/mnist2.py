import itertools
import bz2
import pickle
import keras
import keras.models
import keras.layers
import keras.callbacks
import keras.regularizers
import keras.layers.advanced_activations
import keras.utils.np_utils
import keras.datasets
import sklearn.metrics
import keras.preprocessing.image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys

import numpy as np

if __name__ == '__main__':
    main()

def compute_num_layers_without_stride(input_size, convolution_size):
    n = 0

    while input_size[0] > 2 * (convolution_size[0] // 2) and input_size[1] > 2 * (convolution_size[1] // 2):
        n += 1
        input_size = input_size[0] - 2 * (convolution_size[0] // 2), input_size[1] - 2 * (convolution_size[1] // 2)

    return n, input_size

def create_model_architecture(item_shape, loss_function):
    print('Creating network architecture')

    conv_layer_args = {
        'data_format': 'channels_last',
        'kernel_size': (3, 3),
    }

    norm_layer_args = {
        'axis': 3,
        'momentum': 0.99,
        'epsilon': 0.001,
        'center': True,
        'scale': True,
    }
    
    # Very first layer is always batch normalization
    input_layer = keras.layers.Input(shape=(*item_shape, 1), name='input')
    normalized_input = keras.layers.BatchNormalization(**norm_layer_args,
                                                       name='normalized_input')(input_layer)

    # Want to build the deepest possible representation
    num_representation_layers, representation_shape = compute_num_layers_without_stride(item_shape, (3,3))
    print('Number of representation layers: {} // shape: {}'.format(num_representation_layers, representation_shape))

    current_input = normalized_input
    
    for i in range(num_representation_layers):
        convolve = keras.layers.Conv2D(filters=64 * (4 * (i + 1)), **conv_layer_args)(current_input)
        normalize = keras.layers.BatchNormalization(**norm_layer_args)(convolve)
        activation = keras.layers.PReLU()(normalize)
        current_input = keras.layers.SpatialDropout2D(rate=0.1, data_format='channels_last')(activation)

    # Final layer of representation -- lots of 1d big convolutions
    convolve = keras.layers.Conv2D(filters=2048, kernel_size=(1,1), data_format='channels_last', name='Representation')(current_input)
    pooled = keras.layers.GlobalMaxPooling2D(data_format='channels_last')(convolve)
    
    print('Representation layer shape: {}'.format(pooled.shape))

    # Two hidden layers
    normalized = keras.layers.BatchNormalization()(pooled)
    hidden = keras.layers.Dense(512, use_bias=True)(normalized)
    activation = keras.layers.PReLU()(hidden)
    dropout = keras.layers.Dropout(0.5)(activation)

    normalized = keras.layers.BatchNormalization()(dropout)
    hidden = keras.layers.Dense(128, use_bias=True)(normalized)
    activation = keras.layers.PReLU()(hidden)
    dropout = keras.layers.Dropout(0.5)(activation)

    # Output classification layer
    classification = keras.layers.Dense(10, activation='softmax', use_bias=True)(dropout)

    model = keras.models.Model(inputs=input_layer, outputs=classification)
    model.compile(loss=loss_function,
                  optimizer='nadam',
                  metrics=[
                      'categorical_accuracy',
                  ])

    return model

def load_dataset(batch_size):
    print('Loading MNIST data set')
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    print('Reshaping and converting to float')
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)).astype(np.float32)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)).astype(np.float32)
    
    print('Converting labels to categorical')
    y_train = keras.utils.np_utils.to_categorical(y_train, 10)
    y_test = keras.utils.np_utils.to_categorical(y_test, 10)

    datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=20,
                                                           width_shift_range=0.1,
                                                           height_shift_range=0.1,
                                                           zoom_range=0.1,
                                                           shear_range=5,
                                                           validation_split=0.25)

    train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)
    validation_generator = datagen.flow(x_train, y_train, batch_size=batch_size, subset='validation')

    return train_generator, validation_generator, x_test, y_test, x_train.shape[0]


def perform_evaluation(model, history, x_test, y_test):
    print('Performing evaluation')
    predictions = model.predict(x_test)
    predicted = predictions.argmax(axis=1)
    expected = y_test.argmax(axis=1)

    print('Shapes: predicted {}, expected {}'.format(predicted.shape, expected.shape))

    with PdfPages('analysis.pdf') as pdf:
        print(history.history.keys())
        print('Plotting learning curves')
        f = plt.figure(figsize=(11, 8.5), dpi=600)
        a = f.add_subplot(2, 1, 1)
        a.set_title('Accuracy')
        a.plot(history.history['categorical_accuracy'], label='Training')
        a.plot(history.history['val_categorical_accuracy'], label='Validation')
        a.set_ylabel('Accuracy')
        a.set_xlabel('Epoch')
        a.legend()

        a = f.add_subplot(2, 1, 2)
        a.set_title('Loss')
        a.plot(history.history['loss'], label='Training')
        a.plot(history.history['val_loss'], label='Validation')
        a.set_ylabel('Loss')
        a.set_xlabel('Epoch')
        a.legend()

        f.tight_layout()
        pdf.savefig(f)
        plt.close(f)

        print('Computing evaluation metrics (confusion matrix)')
        confusion = sklearn.metrics.confusion_matrix(expected, predicted)
        f = plt.figure(figsize=(11, 8.5), dpi=600)
        a = f.gca()
        cm = plt.cm.Blues
        src = a.imshow(confusion, cmap=cm)
        f.colorbar(src)

        nf = confusion.astype(np.float32) / confusion.sum(axis=1)[:, np.newaxis]
        t = nf.max() / 2
        for i, j in itertools.product(range(nf.shape[0]), range(nf.shape[1])):
            a.text(j, i, '{}\n{:.02f}%'.format(confusion[i, j], 100 * nf[i, j]), horizontalalignment='center',
                   color='white' if nf[i, j] > t else 'black')

        a.set_title('Confusion matrix')
        a.set_xlabel('True label')
        a.set_ylabel('Predicted label')
        f.tight_layout()
        pdf.savefig(f)
        plt.close(f)

        f = plt.figure(figsize=(11, 8.5), dpi=600)
        num_rows = 4
        num_cols = 3

        class_to_curves = {}
        
        for i in range(10):
            a = f.add_subplot(num_rows, num_cols, i + 1)
            print('Computing ROC for class [{}]'.format(i))
            scores = predictions[:, i].ravel()
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(expected == i, scores)
            tnr = 1 - fpr
            f1 = 2 * tpr * tnr / (sys.float_info.min + tpr + tnr)
            idx = f1.argmax()
            thresh = thresholds[idx]
            peak_f1 = f1[idx]

            class_to_curves[i] = (tpr, tnr, fpr, f1, thresholds)
            
            auc = sklearn.metrics.roc_auc_score(expected == i, scores)

            a.set_title('ROC {}, AUC={:.04f}, F1={:.03f}@{:.03f}'.format(i, auc, peak_f1, thresh))
            a.plot(fpr, tpr, label='ROC')
            a.set_xlabel('False positive rate')
            a.set_ylabel('True positive rate')

        f.tight_layout()
        pdf.savefig(f)
        plt.close(f)

        f = plt.figure(figsize=(11, 8.5), dpi=600)

        for i in range(10):
            (tpr, tnr, fpr, f1, threshold) = class_to_curves[i]
            a = f.add_subplot(num_rows, num_cols, i + 1)
            a.set_title('Class {}'.format(i))
            a.plot(threshold, tpr, label='TPR')
            a.plot(threshold, tnr, label='TNR')
            a.plot(threshold, f1, label='F1')
            a.legend()
            a.set_xlabel('Threshold (model score)')
            a.set_label('Metric value')

        f.tight_layout()
        pdf.savefig(f)
        plt.close(f)

def main():
    losses = ['categorical_crossentropy',
              'sparse_categorical_crossentropy',
              'kullback_leibler_divergence',
              'categorical_hinge']

    batch_size = 1024
    training_generator, validation_generator, x_test, y_test, num_train_samples = load_dataset(batch_size)

    evaluation_results = {}
    
    for l in losses:
        print('Creating model for loss [{}]'.format(l))
        m = create_model_architecture((28, 28), l)

        print('Training model for loss [{}]'.format(l))
        history = m.fit_generator(training_generator,
                                  steps_per_epoch = num_train_samples // batch_size,
                                  epochs = epochs,
                                  verbose = 2,
                                  callbacks = [],
                                  validation_data = validation_generator,
                                  validation_steps = 10,
                                  shuffle = True)

        print('Saving model with loss [{}]'.format(l))
        keras.models.save('mnist2-{}.model'.format(l))

        print('Saving training history for loss [{}]'.format(l))
        with bz2.open('{}.train_history'.format(l), 'wb', compresslevel=9) as file:
            pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)
