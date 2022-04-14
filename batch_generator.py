"""
    Batch Generator for Training
"""

from logging import info, basicConfig, INFO
basicConfig(filemode='info.log', level=INFO, format='%(levelname)s:%(message)s')
from typing import Tuple, List, Generator, Dict
from sequenceConstructor import SequenceConstructor
from numpy import ndarray, array, asarray, zeros, ones, hstack, reshape,transpose, uint8
from cython_modules.data_augmentation.imageDataGenerator import ImageDataGenerator
from cython_modules.data_augmentation.augmentation_manager import  PyImageDataGenerator
from sys import exit
def batch_generator(dataset_size: int, batch_size: int, init_state_size:Tuple[int,int], n_steps: int, features: ndarray,
                    labels: ndarray, locations:ndarray, augment:Tuple, scale:float, normalize:bool, mean:float, std:float,
                    mode:int, mode2:int, mode3:int, model_id:int, glimpse_size:Tuple[int,int], zoom:float):

    state_size_1, state_size_2 = init_state_size

    indices: SequenceConstructor = SequenceConstructor(dataset_size, n_steps, n_steps);
    inputs: Dict[str, ndarray] = None;
    outputs: Dict[str, ndarray] = None;

    # iterate over the minibatches
    i: int = 0
    while True:

        # select the sample indices
        start = batch_size*i
        end = batch_size*(i+1)

        # prepare the minibatch

        # input image

        if(scale!=1 and scale!=0):
            I = indices.samples(features, start, end)/scale;
        if (normalize is True):
            I = (indices.samples(features, start, end)-mean)/std;
        else:
            I = indices.samples(features, start, end)

        # transformation matrix with zoom paramters set to 1
        A = zeros((batch_size, 6), dtype='float32')
        A[:, (0,4)] = 1
        # initial RNN states
        S1:zeros = zeros((batch_size, state_size_1), dtype='float32')
        S2:zeros = zeros((batch_size, state_size_2), dtype='float32')
        # biases
        B1, B2, B3, B4, B5, B6 = None, None, None, None, None, None #type: ndarray, ndarray, ndarray, ndarray, ndarray, ndarray
        if(glimpse_size==(26,26)):
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
        Y_loc, Y_cla = None, None; #type:  ndarray, ndarray

        # labels
        Y_cla = indices.labels(labels, start, end) # sequenz_size x batch_size x categories -> 40 x 2 x 6

        if(zoom==1):
            # location matrix could be not defined
            if(locations is None):
                Y_loc = zeros((batch_size, 6), dtype='float32')
                Y_loc[:,(0,4)]=zoom
                #for j in range(n_steps):
                #    Y_loc[:,j, (0, 4)] = zoom
            else:
                Y_loc = indices.samples(locations, start, end);
        else:
            # localization matrix is definitively not defined N x 10 x 6
            if (locations is None):
                Y_loc = zeros((batch_size, 6), dtype='float32')
                Y_loc[:,(0,4)] =zoom
                #for k in range(n_steps):
                    #Y_loc[:, k, (0, 4)] = zoom

        # when using all outputs for training
        if (mode is not None):
            Y_loc = reshape(Y_loc, (batch_size,1,6))
            Y_loc = hstack([Y_loc for i in range(0, n_steps+mode2)])

            if ((n_steps>1 and not mode3) and (Y_cla.ndim==2)):
                Y_cla = reshape(Y_cla, (batch_size,1,Y_cla.shape[1]))
                Y_cla = hstack([Y_cla for i in range(0, n_steps)])

        i +=1
        if (batch_size*(i+1) > len(indices.getMatrix())):
            i = 0

        #print('VARIABLE "I": {}, VALUE:{}'.format(type(I), I.shape))
        if augment is not None:
            rotation = augment[0]
            dataset_id = augment[1]
            if(I.shape[-1]<160):
                I = transpose(I, (0, 1, 4, 2, 3))
            I = asarray(I, dtype=uint8)
            if (I.shape[-1] == 100):
                dataGenerator = PyImageDataGenerator(I, 30.0, 88, 98, 2.0, 2, 70, 40.0)
            else:
                dataGenerator = PyImageDataGenerator(I, 30.0, 98, 108, 2.0, 2, 70, 40.0)
            I = transpose(dataGenerator.getAugmentedImages(),(0, 1, 3, 4, 2))

        else:
            if (I.shape[2] == 56 and I.shape[3] == 120):
                I = transpose(I, (0, 1, 3, 4, 2))
                # print("from 5 dim (N, S, 7, 8, 120, 160) will be :", I.shape)
            elif (I.shape[2] == 7 and I.shape[3] == 120):
                I = transpose(I, (0, 1, 3, 4, 2))
                # print("from 4 dim (N, S, 7, 120, 160) will be : ", I.shape)
            elif (I.shape[2] == 100 and I.shape[3] == 100):
                pass
        if(model_id==1 or model_id==2):
            inputs = {'input_image': I, 'input_matrix': A,
                      'initial_hidden_state_1': S1, 'initial_cell_state_1': S1,
                      'initial_cell_state_2': S2,
                      'b26': B1, 'b24': B2, 'b12': B3, 'b8': B4, 'b6': B5, 'b4': B6};
            outputs = {'classifications': Y_cla, 'localisations': Y_loc};

        elif model_id==3 or model_id==4:
            inputs = {'input_image': I}
            outputs = {'classifications': Y_cla}

        yield inputs, outputs;


"""
            dataGenerator = ImageDataGenerator(
                                            images=I, rotation_range=int(20 * rotation), horizontal_flip=True if dataset_id > 1 else False,
                                            vertical_flip=0.1, shear_range = (0.20 - 0.10 if dataset_id > 1 else 0) * rotation,
                                            zoom_range = (0.10 + 0.10 if dataset_id > 1 else 0) * rotation, noise_range=20,bright_range=2.0
                                        )
            I = dataGenerator.get_img_array()

            """