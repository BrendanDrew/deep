import itertools
import keras
import keras.models
import keras.layers
import keras.regularizers
import keras.layers.advanced_activations
import keras.utils.np_utils
import keras.datasets
import sklearn.metrics
import matplotlib.pyplot as plt

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


    num_representation_layers = 2

    print('Setting up model architecture')
    model = keras.Sequential()
    model.add(keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, input_shape=(x_test.shape[1], x_test.shape[2], 1)))
    model.add(keras.layers.Conv2D(32, (3, 3), kernel_regularizer=keras.regularizers.l1(1), use_bias=True))
    model.add(keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.SpatialDropout2D(0.25))

    for i in range(num_representation_layers):
        model.add(keras.layers.Conv2D(32, (3, 3), kernel_regularizer=keras.regularizers.l1(1), use_bias=True))
        model.add(keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True))
        model.add(keras.layers.LeakyReLU())
        model.add(keras.layers.SpatialDropout2D(0.25))

    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation='softmax', use_bias=True, kernel_regularizer=keras.regularizers.l1(1)))

    print('Compiling model')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])

    print('Fitting model')
    model.fit(x_train, y_train, batch_size=512, epochs=50, verbose=1)

    print('Performing evaluation')
    predicted = model.predict(x_test).argmax(axis=1)
    expected = y_test.argmax(axis=1)

    print('Shapes: predicted {}, expected {}'.format(predicted.shape, expected.shape))

    print('Computing evaluation metrics')
    confusion = sklearn.metrics.confusion_matrix(expected, predicted)
    f = plt.figure(figsize=(11, 8.5), dpi=600)
    a = f.gca()
    src = a.imshow(confusion, cmap=plt.cm.Blues)
    f.colorbar(src)

    for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
        a.text(j, i, str(confusion[i, j]), horizontalalignment='center', color='red')

    a.set_title('Confusion matrix')
    a.set_xlabel('True label')
    a.set_ylabel('Predicted label')
    f.tight_layout()
    plt.savefig('confusion.pdf', dpi='figure', format='pdf')
    plt.close(f)
