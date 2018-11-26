import itertools
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


def main():
    print('Loading MNIST data set')
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    print('Reshaping and converting to float')
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)).astype(np.float32)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)).astype(np.float32)
    
    print('Converting labels to categorical')
    y_train = keras.utils.np_utils.to_categorical(y_train, 10)
    y_test = keras.utils.np_utils.to_categorical(y_test, 10)

    num_representation_layers = 10
    regularization = 0.0001

    model = create_model_architecture(num_representation_layers, regularization, x_test)

    model.summary()

    print('Creating training data augmentation')
    datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, shear_range=5, validation_split=0.25)

    print('Fitting model')
    batch_size = 512
    epochs = 10000
    train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)
    validation_generator = datagen.flow(x_train, y_train, batch_size=batch_size, subset='validation')
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  validation_data=validation_generator,
                                  validation_steps=x_train.shape[0] // (5 * batch_size),
                                  epochs=epochs,
                                  verbose=1,
                                  callbacks=[
                                      keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy',
                                                                    verbose=1,
                                                                    mode='max',
                                                                    patience=10,
                                                                    restore_best_weights=True),
                                  ])

    perform_evaluation(model, history, x_test, y_test)


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

        
            

def create_model_architecture(num_representation_layers, regularization, x_test):
    print('Setting up model architecture')
    model = keras.Sequential()
    model.add(keras.layers.BatchNormalization(axis=2, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                              input_shape=(x_test.shape[1], x_test.shape[2], 1)))
    model.add(keras.layers.Conv2D(8, (3, 3), use_bias=True, data_format='channels_last'))
    model.add(keras.layers.BatchNormalization(axis=2, momentum=0.99, epsilon=0.001, center=True, scale=True))
    model.add(keras.layers.ReLU())

    if regularization > 0:
        model.add(keras.layers.ActivityRegularization(l1=regularization))
        
    model.add(keras.layers.SpatialDropout2D(0.25))

    for i in range(num_representation_layers):
        model.add(keras.layers.Conv2D(8, (3, 3), use_bias=True, data_format='channels_last'))
        model.add(keras.layers.BatchNormalization(axis=2, momentum=0.99, epsilon=0.001, center=True, scale=True))
        model.add(keras.layers.ReLU())
        if regularization > 0:
            model.add(keras.layers.ActivityRegularization(l1=regularization))
            
        model.add(keras.layers.SpatialDropout2D(0.25))

    model.add(keras.layers.Conv2D(8, (5, 5), use_bias=True, data_format='channels_last'))
    model.add(keras.layers.BatchNormalization(axis=2, momentum=0.99, epsilon=0.001, center=True, scale=True))
    model.add(keras.layers.ReLU())

    if regularization > 0:
        model.add(keras.layers.ActivityRegularization(l1=regularization))

    print('Representation shape: {}'.format(model.get_layer(index=-1).output_shape))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(16, use_bias=True))
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation='softmax', use_bias=True))
    print('Compiling model, lambda = {}'.format(regularization))
    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['categorical_accuracy'])

    keras.utils.plot_model(model, to_file='mnist-model.pdf', show_shapes=True, show_layer_names=True)

    return model
