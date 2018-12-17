import keras
import keras.models
import keras.layers
import keras.utils
import keras.callbacks
import keras.regularizers
import keras.layers.advanced_activations
import keras.utils.np_utils
import keras.preprocessing.image
import random
import tarfile
from PIL import Image
import numpy as np
import math
import sys
import tqdm
import gc

if __name__ == '__main__':
    main()


def generate_samples_end_to_end(filename, patch_size, seed, batch_size=100):
    prng = random.Random()
    prng.seed(seed)

    batch = ([], [])
    with tarfile.open(filename, 'r:*') as tar:
        entries = [x.name for x in tar.getmembers() if x.isreg()]

        while True:
            random.shuffle(entries)

            for thing in entries:
                with tar.extractfile(thing) as jpeg:
                    image = Image.open(jpeg)
                    image.load()
                    try:
                        pix = np.array(image)

                        dynamic_range_factor = 0.99
                        offset = (1 - dynamic_range_factor) / 2

                        if 3 == len(pix.shape) and 3 == pix.shape[2] and pix.shape[0] > patch_size[0] and pix.shape[1] > patch_size[1]:

                            for i in range(10):
                                offset_row = math.floor(prng.uniform(0, pix.shape[0] - patch_size[0]))
                                offset_col = math.floor(prng.uniform(0, pix.shape[1] - patch_size[1]))
                                chip = dynamic_range_factor * pix[offset_row:(offset_row + patch_size[0]), offset_col:(offset_col + patch_size[1]), :].copy().astype(np.float32) / 255 + offset

                                if (patch_size[0], patch_size[1], 3) == chip.shape:
                                    batch[0].append(chip)
                                    batch[1].append(chip)

                                    if len(batch[0]) == batch_size:
                                        yield np.stack(batch[0], axis=0), np.stack(batch[1], axis=0)
                                        batch[0].clear()
                                        batch[1].clear()
                                        gc.collect()

                        del pix
                    finally:
                        image.close()
                        del image

def generate_sample_for_mini_model(encoder_model, baseline_generator):
    if encoder_model is None:
        for x in baseline_generator:
            yield x
    else:
        for initial_samples in baseline_generator:
            transformed = encoder_model.predict_on_batch(initial_samples[0])
            yield (transformed, transformed)


BATCH_SIZE = 500
LAYER_PAIR_TOTAL_SAMPLES = 1250000
LAYER_PAIR_EPOCHS = 50
LAYER_PAIR_SAMPLES_PER_EPOCH = math.ceil(LAYER_PAIR_TOTAL_SAMPLES / LAYER_PAIR_EPOCHS)
LAYER_PAIR_STEPS_PER_EPOCH = math.ceil(LAYER_PAIR_SAMPLES_PER_EPOCH / BATCH_SIZE)


END_TO_END_TOTAL_SAMPLES = 5000000
END_TO_END_EPOCHS = 20
END_TO_END_SAMPLES_PER_EPOCH = math.ceil(END_TO_END_TOTAL_SAMPLES / END_TO_END_EPOCHS)
END_TO_END_STEPS_PER_EPOCH = math.ceil(END_TO_END_SAMPLES_PER_EPOCH / BATCH_SIZE)

FINAL_STEPS_PER_EPOCH = 5 * END_TO_END_STEPS_PER_EPOCH
FINAL_EPOCHS = 50 * END_TO_END_EPOCHS

def train_layer_pair(baseline_generator, chip_size, idx, previous_encoder_network, previous_encoder_size, size):
    if idx > 0:
        pair_input = keras.layers.Input(shape=(previous_encoder_size,))
        model_input = pair_input
    else:
        model_input = keras.layers.Input(shape=chip_size)
        pair_input = keras.layers.Flatten()(model_input)
    new_encoder = keras.layers.Dense(size, activation='relu', use_bias='True')
    dropout = keras.layers.Dropout(0.1)
    normalize = keras.layers.BatchNormalization()
    new_decoder = keras.layers.Dense(previous_encoder_size, activation='sigmoid', use_bias='True')
    decoder_output = new_decoder(normalize(dropout(new_encoder(pair_input))))
    if 0 == idx:
        decoder_output = keras.layers.Reshape(chip_size)(decoder_output)
    mini_model = keras.Model(inputs=model_input, outputs=decoder_output)
    mini_model.compile(optimizer='rmsprop', loss='mse' if idx > 0 else 'binary_crossentropy')

    mini_model_data = generate_sample_for_mini_model(previous_encoder_network, baseline_generator)

    with tqdm.trange(LAYER_PAIR_EPOCHS, leave=False) as epochs:
        for e in epochs:
            for _ in tqdm.trange(LAYER_PAIR_STEPS_PER_EPOCH, leave=False, desc='                     Step'):
                loss = mini_model.train_on_batch(*next(mini_model_data))

            epochs.set_description('Epoch {} / loss {:.05f}'.format(e, loss))

    return new_decoder, new_encoder


def main():
    patch_size = (16, 16)
    chip_size = (*patch_size, 3)

    representation_size = 8
    encoder_layers = 7
    encoder_layer_sizes = [math.ceil(math.exp(math.log(chip_size[0]) + math.log(chip_size[1] + math.log(chip_size[2]) - i * math.log(representation_size)))) for i in range(encoder_layers + 1)]

    print('Layer sizes: [{}]'.format([str(x) for x in encoder_layer_sizes]))

    previous_end_to_end_network = None
    previous_encoder_network = None
    previous_encoder_stack = []
    previous_decoder_stack = []
    previous_encoder_size = np.prod(chip_size)

    end_to_end_generator = generate_samples_end_to_end(sys.argv[1], patch_size, random.getrandbits(64), BATCH_SIZE)

    for (idx, size) in enumerate(encoder_layer_sizes):
        # Model #1: simple, two layer, trained greedily using data generated by the previous encoder network
        new_decoder, new_encoder = train_layer_pair(end_to_end_generator, chip_size, idx, previous_encoder_network,
                                                    previous_encoder_size, size)

        # Model #2: refined end-to-end
        encoder_output, end_to_end_input, end_to_end_model, new_decoder_stack, new_encoder_stack = refine_end_to_end(
            chip_size, end_to_end_generator, new_decoder, new_encoder, previous_decoder_stack, previous_encoder_stack)

        # Track our state thus far
        previous_encoder_network = keras.Model(inputs=end_to_end_input, outputs=encoder_output)
        previous_end_to_end_network = end_to_end_model
        previous_encoder_stack = new_encoder_stack
        previous_decoder_stack = new_decoder_stack
        previous_encoder_size = size


    print('Pretraining phase complete. Now doing the actual heavy-lifting')
    m = previous_end_to_end_network


    end_to_end_model.fit_generator(
        generator=end_to_end_generator,
        steps_per_epoch=FINAL_STEPS_PER_EPOCH,
        epochs=FINAL_EPOCHS,
        verbose=1,
        validation_data=end_to_end_generator,
        validation_steps=max(1, FINAL_STEPS_PER_EPOCH // 100),
        use_multiprocessing=False,
        workers=1)

    m.save('autoencoder.model', overwrite=True, include_optimizer=True)


def refine_end_to_end(chip_size, end_to_end_generator, new_decoder, new_encoder, previous_decoder_stack,
                      previous_encoder_stack):
    # Now that we have the mini model, let's see what we need to do to start re-building our end-to-end network
    end_to_end_input = keras.layers.Input(chip_size)
    current_input = keras.layers.Flatten()(end_to_end_input)
    new_encoder_stack = []
    for x in previous_encoder_stack:
        encoder = keras.layers.Dense.from_config(x.get_config())
        current_input = encoder(current_input)
        encoder.set_weights(x.get_weights())
        new_encoder_stack.append(encoder)
        current_input = keras.layers.Dropout(0.1)(current_input)
        current_input = keras.layers.BatchNormalization()(current_input)
    encoder = keras.layers.Dense.from_config(new_encoder.get_config())
    current_input = encoder(current_input)
    encoder_output = current_input
    encoder.set_weights(new_encoder.get_weights())
    new_encoder_stack.append(encoder)
    new_decoder_stack = []
    decoder = keras.layers.Dense.from_config(new_decoder.get_config())
    current_input = decoder(current_input)
    decoder.set_weights(new_decoder.get_weights())
    new_decoder_stack.append(decoder)
    current_input = keras.layers.Dropout(0.1)(current_input)
    current_input = keras.layers.BatchNormalization()(current_input)
    for x in previous_decoder_stack:
        decoder = keras.layers.Dense.from_config(x.get_config())
        current_input = decoder(current_input)
        decoder.set_weights(x.get_weights())
        new_decoder_stack.append(decoder)
        current_input = keras.layers.Dropout(0.1)(current_input)
        current_input = keras.layers.BatchNormalization()(current_input)
    end_to_end_output = keras.layers.Reshape(chip_size)(current_input)
    end_to_end_model = keras.Model(inputs=end_to_end_input, outputs=end_to_end_output)
    end_to_end_model.compile('rmsprop',
                             loss='binary_crossentropy',
                             metrics=['mse', 'mae', 'mape'])
    # Do fine-tuning of the new end-to-end network
    end_to_end_model.fit_generator(
        generator=end_to_end_generator,
        steps_per_epoch=END_TO_END_STEPS_PER_EPOCH,
        epochs=END_TO_END_EPOCHS,
        verbose=1,
        use_multiprocessing=False,
        workers=1)
    return encoder_output, end_to_end_input, end_to_end_model, new_decoder_stack, new_encoder_stack
