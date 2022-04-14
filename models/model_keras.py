
"""

    Keras implementation of the EDRAM network of Ablavatski et al. (2017)

        * tedram_model      |> tEDRAM (with separate batch normalization per time step)

"""

from typing import List, Tuple, Any
from numpy import ndarray, array, zeros, linspace, asarray, sqrt
from tensorflow.keras.layers import (Input,
                          LSTM,
                          Dense,
                          Activation,
                          Flatten,
                          Reshape,
                          Conv2D,
                          LocallyConnected2D,
                          MaxPooling2D,
                          BatchNormalization,
                          Dropout,
                          concatenate,
                          multiply,
                          add,
                          average,
                          maximum, Lambda)


from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
#from models.spatial_transformation.models.layers_tfw import BilinearInterpolation
#from models.spatial_transformation.models.bilinearInterpolation import BilinearInterpolation
from models.spatial_transformation.models.layers_v2 import BilinearInterpolation
from models.weighted_losses import weighted_mean_squared_error
from tensorflow import Tensor
from .model_keras_cell import tedram_cell
from .slices import Slice
from typing import List, Tuple, Callable, TypeVar,Any

localisation_weights: List[ndarray] = []
localisation_weights.append(array([1.00, 0.25, 1.00, 0.25, 1.00, 1.00]))
localisation_weights.append(array([1.00, 0.00, 0.00, 0.00, 1.00, 0.00]))


def emission_weights(output_size: int, zoom_bias: int) -> List[ndarray]:
    """
        Initialization weights for the Spatial Transformer
        (all weights and biases zero except zoom biases which should be set to 1)

        Parameters:

            * output_size: number of rows
            * zoom_bias: ~

        Returns:

            * weight matrix with zeros and biases [zoom_bias,0,0,0,zoom_bias,0]

    """
    b: ndarray = zeros((2, 3), dtype='float32')
    b[0, 0] = zoom_bias
    b[1, 1] = zoom_bias

    W: ndarray = zeros((output_size, 6), dtype='float32')
    weights: List[ndarray] = [W, b.flatten()]

    return weights


def tedram_model(input_shape:Tuple[int,int,int]=(10,120,160,1), learning_rate:float=0.0001, steps:int=3,
                  glimpse_size:Tuple[int,int]=(26,26), coarse_size:Tuple[int,int]=(12,12),
                  n_filters:int=128, filter_sizes:Tuple[int,int]=(3,5), n_features:int=1024,
                  RNN_size_1:int=512, RNN_size_2:int=512, n_classes:int=4, output_mode:int=0,
                  use_init_matrix:int=True, emission_bias:int=1, clip_value:int=1, unique_emission:int=False,
                  unique_glimpse:int=False, bn:bool=True, dropout:int=0, use_weighted_loss:int=False,
                  localisation_cost_factor:float=1.0) -> Model:
    """
        EDRAM network with temporally separated batch normalization - takes an image and iteratively extracts image
        patches (glimpses) to produce a classification

        Parameters:

            * input_shape: input image dimensions
            * learning_rate: ~
            * steps: number of iterations over the input (or size of the window for sequence processing)
            * glimpse_size: dimensions of the extracted image patch (the glimpse)
            * coarse_size: dimensions of the downscaled image for the initialization of the network
            * hidden_init: the value of the initial hidden state (leads to ValueError --> unused)
            * n_filters: determines the number of filters in the glimpse CNN
            * filter_sizes: dimensions of the glimpse and initialization and CNN kernels
            * n_features: learned features of the glimpse CNN (fc dimension)
            * RNN_size: number of cells in the LSTMs
            * n_classes: number of classes in the output
            * output_mode: 0 only the outputs of the last step are evaluated for the loss or
                           1 outputs of all time steps are concatenated for evaluation
            * use_init_matrix: whether to use the initialization matrix for the first step
            * emission_bias: presets the zoom bias of the emission network
            * clip_value: >0  max value of the zoom factor in the spatial transformer or
                          0   smaller clip values for every step [1.,.85,.70,.55,.40,.25]
            * unique_emission: whether to use a unique emission layer
            * unique_glimpse: whether to use a unique first layer for the glimpse CNN
            * bn: whether to use batch normalization
            * dropout: dropout percentage
            * use_weighted_loss: whether to use weighted versions of the losses, especially class weights
            * localisation_cost_factor: weighting of the localisation cost

        Returns:

            * itself

    """
    filter_size1, filter_size2 = list(zip(filter_sizes, filter_sizes))

    # activate dropout
    do = True if dropout>0 else False

    #######################
    ###  Define Inputs  ###
    #######################

    # input image and localization matrix
    input_image = Input(shape=input_shape, dtype='float32', name='input_image') # 10 x 120 x 160 x 1
    input_matrix = Input(shape=(6,), dtype='float32', name='input_matrix')

    # initial hidden states of the LSTMs
    init_h1 = Input(shape=(RNN_size_1,),  dtype='float32', name='initial_hidden_state_1')
    init_c1 = Input(shape=(RNN_size_1,),  dtype='float32', name='initial_cell_state_1')
    # init_h2 is generated by initialization network
    init_c2 = Input(shape=(RNN_size_2,),  dtype='float32', name='initial_cell_state_2')

    # bias matrices
    if glimpse_size==(26,26):
        b26 = Input(shape=(26,26,1),  dtype='float32', name='b26')
        b24 = Input(shape=(24,24,1),  dtype='float32', name='b24')
        b12 = Input(shape=(12,12,1),  dtype='float32', name='b12')
        b8 = Input(shape=(8,8,1),  dtype='float32', name='b8')
        b6 = Input(shape=(6,6,1),  dtype='float32', name='b6')
        b4 = Input(shape=(4,4,1),  dtype='float32', name='b4')
    else: # glimpse size == (16,16)
        b26 = Input(shape=(16,16,1),  dtype='float32', name='b26')
        b24 = Input(shape=(16,16,1),  dtype='float32', name='b24')
        b12 = Input(shape=(8,8,1),  dtype='float32', name='b12')
        b8 = Input(shape=(8,8,1),  dtype='float32', name='b8')
        b6 = Input(shape=(6,6,1),  dtype='float32', name='b6')
        b4 = Input(shape=(4,4,1),  dtype='float32', name='b4')

    inputs=[input_image, input_matrix, init_h1, init_c1, init_c2, b26, b24, b12, b8, b6, b4]


    #################################
    ###  Network Building Blocks  ###
    #################################
    glimpse_padding:str=None;
    if(glimpse_size==(26,26)):
        glimpse_padding = 'valid'
    else:
        glimpse_padding = 'same'

    ### ------------------------------------------- layers for the EDRAM core cell
    ## Glimpse Network (26x26 --> 192x4x4 --> 1024)
    # 64 filters, 3x3 Convolution, zero padding --> 26x26
    conv_1:Conv2D =None;
    conv_1_bias:LocallyConnected2D=None;
    conv_2:Conv2D=None;
    conv_2_bias:LocallyConnected2D=None;
    max_pooling:MaxPooling2D=None;
    conv_3:Conv2D=None;
    conv_3_bias:LocallyConnected2D = None;
    max_pooling_2:max_pooling =None;
    conv_4:conv_1;
    conv_4_bias:conv_1_bias;
    conv_5:conv_2;
    conv_5_bias:conv_2_bias;
    conv_6:conv_3;
    conv_6_bias:conv_3_bias
    if(unique_glimpse==False):
        conv_1 = Conv2D(int(n_filters/2), filter_size1, padding='same', activation='relu', use_bias=False, name='glimpse_conv1')
        conv_1_bias = LocallyConnected2D(int(n_filters/2), (1,1), padding='valid', use_bias=False, name='glimpse_conv1_bias')
    else:
        conv_1 = None
        conv_1_bias = None
    # 64 filters, 3x3 Convolution, no padding --> 24x24
    conv_2: Conv2D = Conv2D(int(n_filters/2), filter_size1, padding=glimpse_padding, activation='relu', use_bias=False, name='glimpse_conv2')
    conv_2_bias: LocallyConnected2D = LocallyConnected2D(int(n_filters/2), (1,1), padding='valid', use_bias=False, name='glimpse_conv2_bias')
    # max pooling, 24x24 --> 12x12
    max_pooling_1 = MaxPooling2D(pool_size=(2,2), name='glimpse_max_pooling1')
    # 128 filters, 3x3 Convolution, padding to preserve dimensionality of the tensor
    conv_3: Conv2D = Conv2D(n_filters, filter_size1, padding='same', activation='relu', use_bias=False, name='glimpse_conv3')
    conv_3_bias: LocallyConnected2D = LocallyConnected2D(n_filters, (1,1), padding='valid', use_bias=False, name='glimpse_conv3_bias')
    # 128 filters, 3x3 Convolution, padding to preserve dimensionality of the tensor
    conv_4: Conv2D = Conv2D(n_filters, filter_size1, padding='same', activation='relu', use_bias=False, name='glimpse_conv4')
    conv_4_bias: LocallyConnected2D = LocallyConnected2D(n_filters, (1,1), padding='valid', use_bias=False, name='glimpse_conv4_bias')
    # max pooling, 12x12 --> 6x6
    max_pooling_2 = MaxPooling2D(pool_size=(2,2), name='glimpse_max_pooling2')
    # 160 filters, 3x3 Convolution, padding to preserve dimensionality of the tensor
    conv_5 = Conv2D(160, filter_size1, padding='same', activation='relu', use_bias=False, name='glimpse_conv5')
    conv_5_bias = LocallyConnected2D(160, (1,1), padding='valid', use_bias=False, name='glimpse_conv5_bias')
    # 192 filters, 3x3 Convolution, no padding --> 4x4
    conv_6 = Conv2D(192, filter_size1, padding=glimpse_padding, activation='relu', use_bias=False, name='glimpse_conv6')
    conv_6_bias = LocallyConnected2D(192, (1,1), padding='valid', use_bias=False, name='glimpse_conv6_bias')
    # 4*4*192 = 3072
    flatten:Flatten = Flatten(name='glimpse_flatten')

    # fully connected, output_dim=1024
    glimpse_what:Dense = Dense(n_features, activation='relu', name = 'glimpse_what')

    # fully connected, output_dim=1024
    glimpse_where:Dense = Dense(n_features, activation='relu', name = 'glimpse_where')
    # --> glimpse_what and where are multiplied to create Glimpse Network output

    ## LSTMs
    # reshape to 1-step sequence for LSTM
    reshape_to_sequence:Reshape = Reshape((1, n_features), name = 'to_sequence')
    # LSTMs
    LSTM_classify:LSTM = LSTM(RNN_size_1, return_state=True, return_sequences=True, name="LSTM_classify")
    LSTM_localize:LSTM = LSTM(RNN_size_2, return_state=True, name="LSTM_localize")
    # classification network - outputs classification probabilities
    reshape_from_sequence = Reshape((RNN_size_1,), name='from_sequence')

    ## Classification Network
    # fully connected, output_dim=1024
    cla_fc_1 = Dense(n_features, activation='relu', name='classification_fc1')
    # fully connected, output_dim=1024
    cla_fc_2 = Dense(n_features, activation='relu', name='classification_fc2')
    # fully connected, output_dim=7, softmax activation
    cla_fc_3 = Dense(n_classes, activation='softmax', name='classification_fc3')

    ## Emission Network
    if unique_emission==False:
        em = Dense(6, activation='tanh', weights=emission_weights(RNN_size_2, emission_bias), name='emission')
    else:
        em = None

    ## pack layers
    layers = (conv_1, conv_1_bias, conv_2, conv_2_bias, max_pooling_1,
              conv_3, conv_3_bias, conv_4, conv_4_bias, max_pooling_2, conv_5, conv_5_bias,
              conv_6, conv_6_bias, flatten, glimpse_what, glimpse_where,
              reshape_to_sequence, LSTM_classify, LSTM_localize, reshape_from_sequence,
              cla_fc_1, cla_fc_2, cla_fc_3, em)

    ### the EDRAM Core Cells
    output_localisation: bool = False if output_mode==0 and steps==1 else True
    # constant or decreassing clip values for the zoom factor of the glimpse STN
    clip_value: ndarray = linspace(clip_value, clip_value, steps) if clip_value>0 else linspace(1, 0.25, steps)
    # constant or decreassing bias for the zoom factor of the emission network
    emission_bias: ndarray = linspace(emission_bias, emission_bias, steps) if emission_bias>0 else linspace(1, 0.30, steps)
    edram_cell: List[Model] = []
    # STEPS should be set to 10
    for i in range(0, steps):
        print('INPUT SHAPE FROM MODEL TEDRAM2', input_shape)
        print('CALL TEDRAM CELL FROM MODEL TEDRAM2:', input_shape[0:])
        edram_cell.append(tedram_cell(input_shape=input_shape[0:], glimpse_size=glimpse_size, n_filters=n_filters, RNN_size_1=RNN_size_1,
                                       RNN_size_2=RNN_size_2, bn=bn, dropout=dropout, clip_value=clip_value[i], layers=layers,
                                       output_localisation=output_localisation, step=i, unique_emission=unique_emission,
                                       unique_glimpse=unique_glimpse, emission_bias=emission_bias[i]))
    ### Initialization Network
    ### Context Network
    ## downscale the input with STN, TODO: Do this in the preprocessing (but then coarse_size is fixed!)
    slCoarse: Slice = Slice([0, 0, 0, 0, 0], [-1, 1, -1, -1, -1], name='sliceCoarse')
    input_image_co: Tensor = slCoarse(input_image)
    #input_image_coarse: Tensor = BilinearInterpolation(coarse_size, 1)([input_image_co, input_matrix])
    input_image_coarse: Tensor = BilinearInterpolation(height=coarse_size[0], width=coarse_size[1])([input_image_co, input_matrix]);
    print('INPUT IMAGE COARSE:',input_image_coarse.shape) #(?, 12, 12, ?)
    ## 3 convolutions on 1x12x12 input: 5x5, 16 filters --> 3x3, 16 filters --> 3x3, 32 filters
    bn_axis: int = 3
    x = Conv2D(int(n_filters/8), filter_size2, padding='valid', use_bias=False, name='init_conv1')(input_image_coarse)
    b = LocallyConnected2D(int(n_filters/8), (1,1), padding='valid', use_bias=False, name='init_conv1_bias')(b8)
    x = add([x, b], name='init_conv1_add')

    if do: x = Dropout(dropout, name='init_conv1_dropout')(x)
    if bn: x = BatchNormalization(axis=bn_axis, name='init_conv1_bn')(x)

    nf = n_filters/8 if RNN_size_2==512 else n_filters/4 # increasing glimpse NN size if RNN_size2 is 1024
    x = Conv2D(int(nf), filter_size1, padding='valid', use_bias=False, name='init_conv2')(x)
    b = LocallyConnected2D(int(nf), (1,1), padding='valid', use_bias=False, name='init_conv2_bias')(b6)
    x = add([x, b], name='init_conv2_add')

    if do: x = Dropout(dropout, name='init_conv2_dropout')(x)
    if bn: x = BatchNormalization(axis=bn_axis, name='init_conv2_bn')(x)

    nf = n_filters/4 if RNN_size_2==512 else n_filters/2 # increasing glimpse NN size if RNN_size2 is 1024
    x = Conv2D(int(nf), filter_size1, padding='valid', use_bias=False, name='init_conv3')(x)
    b = LocallyConnected2D(int(nf), (1,1), padding='valid', use_bias=False, name='init_conv3_bias')(b4)
    x = add([x, b], name='init_conv3_add')

    if do: x = Dropout(dropout, name='init_conv3_dropout')(x)
    if bn: x = BatchNormalization(axis=bn_axis, name='init_conv3_bn')(x)
    x = Flatten(name='init_flatten')(x)

    ## Initialization of localization LSTM
    if unique_emission==False:
        init_matrix = em(x)
    else:
        init_matrix = Dense(6, activation='tanh', weights=emission_weights(RNN_size_2, emission_bias[0]), name='emission')(x)

    init_h2 = Reshape((RNN_size_2,), name = 'initial_hidden_state_2')(x)

    #############################
    ###  Assemble everything  ###
    #############################
    #input_image0, input_image1 = None, None #type: Lambda, Lambda;	
    input_image1: Tensor
    input_image0: Tensor
    sl0: Slice = Slice([0, 0, 0, 0, 0], [-1, 1, -1, -1, -1], name='sel0')
    # -----------------  step zero (initialization)
    step = [[None, init_matrix if output_mode==1 else input_matrix]]
    # ------------------  step 1, INPUT_IMAGE (None, 10, 120,160,1) -> input_image[0][0] -> (120,160,1)
    #input_image0: Lambda = Lambda(lambda x:x[:,0])(input_image);
    input_image0 = sl0(input_image)
    step.append(edram_cell[0]([input_image0, init_matrix if use_init_matrix else input_matrix, init_h1, init_c1, init_h2, init_c2, b26, b24, b12, b6 if glimpse_size==(26,26) else b4, b4]))
    # -------------------  "recurrently" apply edram network
    for i in range(1, steps):
        sli = Slice([0, i, 0, 0, 0], [-1, 1, -1, -1, -1], name='sel{}'.format(i))
        input_image1 = sli(input_image)
        step.append(edram_cell[i]([input_image1, step[i][1], step[i][2], step[i][3], step[i][4], step[i][5], b26, b24, b12, b6 if glimpse_size==(26,26) else b4, b4]))


    ########################
    ###  Define Outputs  ###
    ########################

    if output_mode==0:
        # only use outputs of last time step
        classifications = Reshape((n_classes,), name='classifications')(step[steps][0])
        localisations = Reshape((6,), name='localisations')(step[steps-1][1])
    else:
        # concatenate outputs of different timesteps
        if steps==1:
            classifications = Reshape((n_classes,), name='classifications')(step[1][0])
            localisations = Reshape((2, 6), name='localisations')(concatenate([step[0][1], step[1][1]]))
        else:
            classifications = Reshape((steps, n_classes), name='classifications')(concatenate([step[i][0] for i in range(1, steps+1)]))
            localisations = Reshape((steps+1, 6), name='localisations')(concatenate([step[i][1] for i in range(0, steps+1)]))

    outputs:List[Any]=[classifications, localisations]
    
    print(classifications.shape)
    print(localisations.shape)
    # build the model
    print(len(inputs), len(outputs))
    model = Model(inputs, outputs, name='tedram_model')

    ############################
    ###  Training Framework  ###
    ############################

    # optimization algorithm
    optimizer = Adam(lr=learning_rate, clipnorm=10.)
    classification_loss:str = ' ';
    localisation_loss:str= ' ';

    # define losses
    if use_weighted_loss:
        # weighted losses
        if n_classes==4:
            # only use collision outcome weights for collision classification
            # classification_loss = weighted_categorical_crossentropy(emotion_weights[use_weighted_loss])
            classification_loss = None;
        else:
            classification_loss = 'categorical_crossentropy'
        localisation_loss = weighted_mean_squared_error(localisation_weights[1 if n_classes==4 else 0])
    else:
        # standard losses
        classification_loss = 'categorical_crossentropy'
        localisation_loss = 'mean_squared_error'
    
    print('classification loss:', classification_loss)
    print('localisation_loss:', localisation_loss)
    ###########################
    ###  Compile the Model  ###
    ###########################

    model.compile(loss={'classifications': classification_loss, 'localisations': localisation_loss},
                  loss_weights={'classifications': 1, 'localisations': localisation_cost_factor},
                  metrics={'classifications': 'categorical_accuracy'}, optimizer=optimizer)

    return model
