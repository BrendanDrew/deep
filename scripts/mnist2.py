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
from keras import backend as K
from matplotlib.backends.backend_pdf import PdfPages
import sys
import gc

import numpy as np

if __name__ == '__main__':
    main()

def print_model_summary(m, file=sys.stdout):
    num_layers = len(m.layers)
    num_parameters = m.count_params()
    print('TOTAL: {} LAYERS, {} PARAMETERS'.format(num_layers, num_parameters), file=file)

def print_model_layer(l, file=sys.stdout):
    input_description = ''

    if hasattr(l, 'inputs'):
        input_description = ', '.join([x.name for x in l.inputs])

    output_description = ''
    if hasattr(l, 'outputs'):
        output_description = ', '.join([x.name for x in l.outputs])

    input_size = ''
    if hasattr(l, 'input_shape'):
        input_size = str(l.input_shape)

    output_size = ''
    if hasattr(l, 'output_shape'):
        output_size = str(l.output_shape)


    layer_attributes = [
        'filters',
        'kernel_size',
        'strides',
        'padding',
        'data_format',
        'dilation_rate',
        'activation',
        'use_bias',
        'units',
    ]

    additional_attributes = {attribute: getattr(l, attribute) for attribute in layer_attributes if hasattr(l, attribute)}

    print('  LAYER: {} (\'{}\') // {} PARAMETERS'.format(type(l).__name__, l.name, l.count_params()), file=file)
    print('    INPUT SHAPE:  {}'.format(input_size), file=file)
    print('    INPUTS:       {}'.format(input_description), file=file)
    print('    OUTPUT SHAPE: {}'.format(output_size), file=file)
    print('    OUTPUTS:      {}'.format(output_description), file=file)
    print('    WEIGHTS:', file=file)

    if hasattr(l, 'trainable_weights'):
        for t in l.trainable_weights:
            print('        T {} {}'.format(t.name, t.shape), file=file)

    if hasattr(l, 'untrainable_weights'):
        for u in l.untrainable_weights:
            print('        U {} {}'.format(u.name, u.shape), file=file)

    print('    ATTRIBUTES:', file=file)

    for (attribute, value) in sorted(additional_attributes.items(), key=lambda x: x[0]):
        print('        {:20}: {}'.format(attribute, value), file=file)


def print_model(m, file=sys.stdout):
    print_model_summary(m, file=file)

    for l in m.layers:
        print_model_layer(l, file=file)


def compute_num_layers_with_stride(input_size, convolution_size, stride):
    n = 0

    while input_size[0] > (convolution_size[0] - 1) and input_size[1] > (convolution_size[1] - 1):
        n += 1
        input_size = ((input_size[0] - convolution_size[0] +1) // stride[0]), ((input_size[1] - convolution_size[1] + 1) // stride[1])

    return n, input_size


def compute_num_layers_without_stride(input_size, convolution_size):
    n = 0

    while input_size[0] > 2 * (convolution_size[0] // 2) and input_size[1] > 2 * (convolution_size[1] // 2):
        n += 1
        input_size = input_size[0] - 2 * (convolution_size[0] // 2), input_size[1] - 2 * (convolution_size[1] // 2)

    return n, input_size


def create_model_architecture(item_shape, loss_function, optimizer, representation_size, convolution_depth):
    print('Creating network architecture')

    convolution_size = (5, 5)
    convolution_stride = (1, 1)

    conv_layer_args = {
        'data_format': 'channels_last',
        'kernel_size': convolution_size,
        'use_bias': True,
        'activation': 'relu',
        'strides': convolution_stride
    }

    norm_layer_args = {
        'axis': 3,
        'momentum': 0.99,
        'epsilon': 0.001,
        'center': True,
        'scale': True,
    }

    dropout_args = {
        'rate': 0.25,
        'data_format': 'channels_last'
    }

    # Very first layer is always batch normalization
    input_layer = keras.layers.Input(shape=(*item_shape, 1), name='input')
    normalized_input = keras.layers.BatchNormalization(**norm_layer_args,
                                                       name='normalized_input')(input_layer)

    # Want to build the deepest possible representation
    num_representation_layers, representation_shape = compute_num_layers_with_stride(item_shape, convolution_size, convolution_stride)
    print('Number of representation layers: {} // shape: {}'.format(num_representation_layers, representation_shape))

    current_input = normalized_input

    for i in range(num_representation_layers):
        current_input = keras.layers.Conv2D(filters=convolution_depth, **conv_layer_args)(current_input)
        current_input = keras.layers.BatchNormalization(**norm_layer_args)(current_input)

    current_input = keras.layers.Conv2D(filters=convolution_depth, kernel_size=representation_shape, data_format='channels_last', name='final_convolution')(current_input)
    current_input = keras.layers.BatchNormalization(**norm_layer_args)(current_input)
    print('After 1D convolutions, shape is [{}] @ representation size {}'.format(current_input.shape, representation_size))

    current_input = keras.layers.Flatten()(current_input)

    print('Representation layer shape: {}'.format(current_input.shape))

    current_input = keras.layers.BatchNormalization()(current_input)

    current_input = keras.layers.Dense(representation_size, activation='relu', use_bias=True, name='hidden')(current_input)
    current_input = keras.layers.BatchNormalization()(current_input)
    current_input = keras.layers.Dropout(0.5)(current_input)

    # Output classification layer
    classification = keras.layers.Dense(10, activation='softmax', use_bias=True, name='output')(current_input)

    model = keras.models.Model(inputs=input_layer, outputs=classification)
    model.compile(loss=loss_function,
                  optimizer=optimizer,
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

    print('Shuffling training data')
    p = np.random.permutation(x_train.shape[0])
    tmp = x_train
    x_train = tmp[p]
    tmp = y_train
    y_train = tmp[p]

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

    confusion = sklearn.metrics.confusion_matrix(expected, predicted)
    nf = confusion.astype(np.float32) / confusion.sum(axis=1)[:, np.newaxis]
    class_to_curves = {}
    class_to_auc = {}

    for i in range(10):
        scores = predictions[:, i].ravel()
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(expected == i, scores)
        tnr = 1 - fpr
        f1 = 2 * tpr * tnr / (sys.float_info.min + tpr + tnr)
        class_to_curves[i] = {
            'true_positive_rate': tpr,
            'true_negative_rate': tnr,
            'false_positive_rate': fpr,
            'f1': f1,
            'thresholds': thresholds
        }

        class_to_auc[i] = sklearn.metrics.roc_auc_score(expected == i, scores)

    accuracy = 100 * np.diag(confusion).sum() / confusion.sum()

    print('Correctly classified: {}, total {}, accuracy: {}%'.format(np.diag(confusion).sum(), confusion.sum(), accuracy))

    return {
        'training_loss': history.history['loss'],
        'validation_loss': history.history['val_loss'],
        'training_accuracy': history.history['categorical_accuracy'],
        'validation_accuracy': history.history['val_categorical_accuracy'],
        'confusion': nf,
        'class_to_curves': class_to_curves,
        'class_to_auc': class_to_auc,
        'accuracy': accuracy,
    }


def figure():
    return plt.figure(figsize=(17, 11), dpi=600)


def plot_learning_curves(loss_to_evaluation, pdf):
    for (loss, evaluation) in sorted(loss_to_evaluation.items(), key=lambda x: x[0]):
        f = figure()
        a = f.add_subplot(1, 2, 1)
        f.suptitle('Learning curves (loss={})'.format(loss))
        a.plot(evaluation['training_loss'], label='Training')
        a.plot(evaluation['validation_loss'], label='Validation')
        a.set_xlabel('Epoch')
        a.set_ylabel('Loss')
        a.set_title('Loss')
        a.legend()

        a = f.add_subplot(1, 2, 2)
        a.plot(evaluation['training_accuracy'], label='Training')
        a.plot(evaluation['validation_accuracy'], label='Validation')
        a.set_xlabel('Epoch')
        a.set_ylabel('Accuracy')
        a.set_title('Accuracy')
        a.legend()

        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(f)
        plt.close(f)


def plot_confusion_matrices(loss_to_evaluation, pdf):
    for (loss, evaluation) in sorted(loss_to_evaluation.items(), key=lambda x:x[0]):
        f = figure()
        a = f.gca()
        cm = plt.cm.Blues
        confusion = evaluation['confusion']
        src = a.imshow(confusion, cmap=cm)
        f.colorbar(src)

        t = confusion.max() / 2
        for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
            a.text(j, i, '{:.02f}%'.format(100 * confusion[i, j]), horizontalalignment='center', color='white' if confusion[i, j] > t else 'black')

        a.set_title('Confusion matrix (loss={}), Accuracy: {:.04f}%'.format(loss, evaluation['accuracy']))
        a.set_xlabel('True label')
        a.set_ylabel('Predicted label')
        pdf.savefig(f)
        plt.close(f)


def plot_roc(loss_to_evaluation, pdf):
    f = figure()
    rows = 3
    cols = 4

    for cls in range(10):
        data = []

        for (loss, evaluation) in loss_to_evaluation.items():
            tpr = evaluation['class_to_curves'][cls]['true_positive_rate']
            fpr = evaluation['class_to_curves'][cls]['false_positive_rate']
            auc = evaluation['class_to_auc'][cls]
            label = '{} ({:.03f} AUC)'.format(loss, auc)
            data.append((label, fpr, tpr))

        a = f.add_subplot(rows, cols, cls + 1)

        for (label, x, y) in sorted(data, key=lambda x: x[0]):
            a.plot(x, y, label=label)

        a.set_xlabel('FPR')
        a.set_ylabel('TPR')
        a.set_title('Class [{}]'.format(cls))
        a.legend()

    f.tight_layout()
    pdf.savefig(f)
    plt.close(f)

    for loss, evaluation in sorted(loss_to_evaluation.items(), key=lambda x: x[0]):
        for cls, curves in sorted(evaluation['class_to_curves'].items(), key=lambda x: x[0]):
            f = figure()
            a = f.gca()
            f.suptitle('Loss: {}'.format(loss))
            tpr = curves['true_positive_rate']
            fpr = curves['false_positive_rate']
            thresholds = curves['thresholds']

            a.plot(thresholds, tpr, label='True positive rate')
            a.plot(thresholds, fpr, label='False positive rate')
            a.set_title('Class {}'.format(cls, loss))
            a.set_xlabel('Threshold')
            a.set_ylabel('Rate')
            a.legend()
            pdf.savefig(f)
            plt.close(f)


def plot_f1(loss_to_evaluation, pdf):
    for loss, evaluation in sorted(loss_to_evaluation.items(), key=lambda x: x[0]):
        for cls, curves in sorted(evaluation['class_to_curves'].items(), key=lambda x: x[0]):
            f = figure()
            a = f.gca()
            f.suptitle('Loss: {}'.format(loss))
            thresholds = curves['thresholds']
            f1 = curves['f1']
            peak_f1 = f1.max()

            a.plot(thresholds, f1, label='F1 (peak={:.04f})'.format(peak_f1))
            a.set_xlabel('Threshold')
            a.set_ylabel('F1')
            a.set_title('Class {} F1'.format(cls))

            pdf.savefig(f)
            plt.close(f)


def generate_plots(loss_to_evaluation):
    with PdfPages('mnist2-analysis.pdf') as pdf:
        plot_learning_curves(loss_to_evaluation, pdf)
        plot_confusion_matrices(loss_to_evaluation, pdf)
        plot_roc(loss_to_evaluation, pdf)
        plot_f1(loss_to_evaluation, pdf)


def main():
    losses = [
        'categorical_crossentropy',
    ]

    loss_short_names = {
        'categorical_crossentropy': 'CC',
        'kullback_leibler_divergence': 'KL-DIV'
    }

    optimizers = [
        'rmsprop',
    ]

    batch_size = 250
    training_generator, validation_generator, x_test, y_test, num_train_samples = load_dataset(batch_size)
    epochs = 500

    loss_to_evaluation = {}

    representation_sizes = [
        64,
    ]

    layer1_sizes = [
        2, 8, 32, 128
    ]

    for l in losses:
        for o in optimizers:
            for representation_size in representation_sizes:
                for layer1_size in layer1_sizes:
                    try:
                        print('Creating model for loss [{}/{}], representation={}, convolution depth={}'.format(l, o, representation_size, layer1_size))
                        m = create_model_architecture((28, 28), l, o, representation_size, layer1_size)
                        m.summary(line_length=120)

                        model_description = '{}:{} ({} / {}: {})'.format(loss_short_names[l], o, representation_size, layer1_size, m.count_params())

                        print('Training model {}'.format(model_description))
                        history = m.fit_generator(training_generator,
                                                  steps_per_epoch=num_train_samples // batch_size,
                                                  epochs=epochs,
                                                  verbose=1,
                                                  callbacks=[
                                                      keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy',
                                                                                    verbose=1,
                                                                                    mode='max',
                                                                                    patience=25,
                                                                                    restore_best_weights=False),
                                                  ],
                                                  validation_data=validation_generator,
                                                  validation_steps=20000 // batch_size,
                                                  shuffle=True)

                        print('Evaluating model'.format(l))
                        loss_to_evaluation[model_description] = perform_evaluation(m, history, x_test, y_test)

                        print('Cleaning up')
                        del m
                        del history
                        K.clear_session()
                        gc.collect()
                    except ValueError:
                        print('Failed to create / train / evaluate model for {} / {} / {}'.format(l, representation_size, layer1_size))

    print('Starting evaluation and plot generation')
    generate_plots(loss_to_evaluation)
