"""
    Batch Generator for Training
"""

from logging import info, basicConfig, INFO
basicConfig(filemode='info.log', level=INFO, format='%(levelname)s:%(message)s')

from typing import Tuple, Dict
try:
    from .sequenceConstructor import SequenceConstructor
except ImportError as I:
    from sequenceConstructor import SequenceConstructor

from numpy import ndarray, array, asarray, zeros, ones, hstack, reshape,transpose, uint8
# from cython_modules.data_augmentation.imageDataGenerator import ImageDataGenerator
try:
    from c_modules.data_augmentation.augmentation_manager import PyImageDataGenerator
except ImportError or ModuleNotFoundError:
    from .c_modules.data_augmentation.augmentation_manager import PyImageDataGenerator

def batch_generator(dataset_size: int, batch_size: int, n_steps: int, n_classes: int, init_state_size: Tuple[int,int], features: ndarray, labels: ndarray, locations: ndarray, augment: bool, scale: float, normalize: bool,
                    mean: float, std: float, mode: int, mode2: int, mode3: int, model_id: int, glimpse_size: Tuple[int,int], zoom: float):

    state_size_1, state_size_2 = init_state_size

    indices: SequenceConstructor = SequenceConstructor(dataset_size, n_steps, n_classes)
    inputs: Dict[str, ndarray] = None
    outputs: Dict[str, ndarray] = None

    # iterate over the minibatches
    i: int = 0
    while True:

        # select the sample indices
        start = batch_size*i
        end = batch_size*(i+1)

        # prepare the minibatch
        # input image

        if scale != 1 and scale != 0:
            I = indices.samples(features, start, end)/scale
        if normalize is True:
            I = (indices.samples(features, start, end)-mean)/std
        else:
            I = indices.samples(features, start, end)

        # transformation matrix with zoom paramters set to 1
        A = zeros((batch_size, 6), dtype='float32')
        A[:, (0,4)] = 1
        # initial RNN states
        S1 = zeros((batch_size, state_size_1), dtype='float32')
        S2= zeros((batch_size, state_size_2), dtype='float32')
        # biases
        if glimpse_size == (26, 26):
            B1 = ones((batch_size, 26, 26, 1), dtype='float32')
            B2 = ones((batch_size, 24, 24, 1), dtype='float32')
            B3 = ones((batch_size, 12, 12, 1), dtype='float32')
            B4 = ones((batch_size, 8, 8, 1), dtype='float32')
            B5 = ones((batch_size, 6, 6, 1), dtype='float32')
            B6 = ones((batch_size, 4, 4, 1), dtype='float32')
        else:
            B1 = ones((batch_size, 16, 16, 1), dtype='float32')
            B2 = ones((batch_size, 16, 16, 1), dtype='float32')
            B3 = ones((batch_size, 8, 8, 1), dtype='float32')
            B4 = ones((batch_size, 8, 8, 1), dtype='float32')
            B5 = ones((batch_size, 6, 6, 1), dtype='float32')
            B6 = ones((batch_size, 4, 4, 1), dtype='float32')

        # target outputs
        # labels
        Y_cla = indices.labels(labels, start, end)  # sequenz_size x batch_size x categories -> 40 x 2 x 6

        if zoom == 1:
            # location matrix could be not defined
            if locations is None:
                Y_loc = zeros((batch_size, 6), dtype='float32')
                Y_loc[:, (0, 4)] = zoom
            else:
                Y_loc = indices.samples(locations, start, end)
        else:
            # localization matrix is definitively not defined N x 10 x 6
            if locations is None:
                Y_loc = zeros((batch_size, 6), dtype='float32')
                Y_loc[:, (0, 4)] = zoom

        # when using all outputs for training
        if mode is True:
            Y_loc = reshape(Y_loc, (batch_size, 1, 6))
            Y_loc = hstack([Y_loc for _ in range(0, n_steps+mode2)])

            if (n_steps > 1 and not mode3) and (Y_cla.ndim == 2):
                Y_cla = reshape(Y_cla, (batch_size,1,Y_cla.shape[1]))
                Y_cla = hstack([Y_cla for _ in range(0, n_steps)])

        i += 1
        if batch_size*(i+1) > len(indices.getMatrix()):
            i = 0

        if augment is True:

            if I.shape[-1] < 160:
                I = transpose(I, (0, 1, 4, 2, 3))

            I = asarray(I, dtype=uint8)

            if I.shape[-1] == 100:
                dataGenerator = PyImageDataGenerator(I, 30.0, 88, 98, 2.0, 2, 70, 40.0)

            else:
                dataGenerator = PyImageDataGenerator(I, 30.0, 90, 100, 2.0, 2, 70, 40.0)

            I = transpose(dataGenerator.getAugmentedImages(), (0, 1, 3, 4, 2))

        else:
            if I.shape[2] == 56 and I.shape[3] == 120:
                I = transpose(I, (0, 1, 3, 4, 2))

            elif I.shape[2] == 7 and I.shape[3] == 120:
                I = transpose(I, (0, 1, 3, 4, 2))

            elif I.shape[2] == 100 and I.shape[3] == 100:
                pass

        if model_id == 1 or model_id == 2:
            inputs = {'input_image': I, 'input_matrix': A,
                      'initial_hidden_state_1': S1, 'initial_cell_state_1': S1,
                      'initial_cell_state_2': S2,
                      'b26': B1, 'b24': B2, 'b12': B3, 'b8': B4, 'b6': B5, 'b4': B6}
            outputs = {'classifications': Y_cla, 'localisations': Y_loc}

        elif model_id == 3 or model_id == 4:
            inputs = {'input_image': I}
            outputs = {'classifications': Y_cla}

        yield inputs, outputs


