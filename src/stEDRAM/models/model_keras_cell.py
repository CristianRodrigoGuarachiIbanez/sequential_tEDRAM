"""

    Keras implementation of the EDRAM network of Ablavatski et al. (2017)

        * tedram_cell       |

"""

from numpy import ndarray, array, zeros, asarray, sqrt
from tensorflow.keras.layers import (Input,
                                     Dense,
                                     Conv2D,
                                     LocallyConnected2D,
                                     BatchNormalization,
                                     Dropout,
                                     multiply,
                                     add)

from tensorflow.keras.models import Model
#from models.spatial_transformation.models.layers_tfw import BilinearInterpolation
#from models.spatial_transformation.models.bilinearInterpolation import BilinearInterpolation
from src.stEDRAM.models.spatial_transformation.models.layers_v2 import BilinearInterpolation
from typing import List,Tuple, TypeVar, Any
from tensorflow import Tensor
# number of emotions per class in ANet training file (in thousands)
n: ndarray = asarray([78, 144, 29, 16, 8, 5, 28])
w: float = sum(n)/(n*7)
w2: ndarray = sqrt(w)


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


def tedram_cell(input_shape:Tuple[int,int,int]=(120,160, 1), glimpse_size:Tuple[int,int]=(26,26),
                 n_filters:int=128, RNN_size_1:int=512, RNN_size_2:int=512, bn:bool=True, dropout:int=0, clip_value:int=1,
                 layers:Tuple=None, output_localisation:bool=True, step:int=0, unique_emission:int=False,
                 unique_glimpse:int=False, emission_bias:int=1) -> Model:
    """
        One timestep of the EDRAM network with temporally separated batch normalization

        Parameters:

            * input_shape: input image dimensions
            * glimpse_size: dimensions of the extracted image patch (the glimpse)
            * n_filters: determines the number of filters in the glimpse CNN
            * filter_size: dimensions of the glimpse CNN kernel
            * n_features: learned features of the glimpse CNN (fc dimension)
            * RNN_size: number of cells in the LSTMs
            * n_classes: number of classes in the output
            * bn: whether to use batch normalization
            * dropout: dropout percentage
            * network: all layers that should be reused
            * output_localisation: whether to output the localisation matrix
            * output_emotion_dims: whether to output valence and arousal for emotion stimuli
            * step: name suffix for the edram cell
            * unique_emission: whether to use a temporally separated emission layer
            * unique_glimpse: whether to use a temporally separated first layer for the glimpse CNN
            * emission_bias: presets the zoom bias of the emission network
            * clip_value: max value of the zoom factor in the spatial transformer

        Returns:

            * itself

    """
    # activate dropout
    do: bool = True if dropout>0 else False

    # unpack layers
    (conv_1, conv_1_bias, conv_2, conv_2_bias, max_pooling_1, conv_3, conv_3_bias, conv_4, conv_4_bias, max_pooling_2, conv_5,
     conv_5_bias, conv_6, conv_6_bias, flatten, glimpse_what, glimpse_where, reshape_to_sequence, LSTM_classify, LSTM_localize,
     reshape_from_sequence, cla_fc_1, cla_fc_2, cla_fc_3, em) = layers

    #######################
    ###  Define Inputs  ###
    #######################

    # input image and localization matrix
    input_image: Input = Input(shape=input_shape, dtype='float32', name='input_image')
    input_matrix: Input = Input(shape=(6,), dtype='float32', name='input_matrix')

    # hidden states of the LSTMs
    hidden_state_1: Input = Input(shape=(RNN_size_1,),  dtype='float32', name='hidden_state_1')
    cell_state_1: Input = Input(shape=(RNN_size_1,),  dtype='float32', name='cell_state_1')
    hidden_state_2: Input = Input(shape=(RNN_size_2,),  dtype='float32', name='hidden_state_2')
    cell_state_2: Input = Input(shape=(RNN_size_2,),  dtype='float32', name='cell_state_2')

    # bias matrices
    if (glimpse_size==(26,26)):
        bias_26: Input = Input(shape=(26,26,1),  dtype='float32', name='bias_26')
        bias_24: Input = Input(shape=(24,24,1),  dtype='float32', name='bias_24')
        bias_12: Input = Input(shape=(12,12,1),  dtype='float32', name='bias_12')
        bias_6: Input = Input(shape=(6,6,1),  dtype='float32', name='bias_6')
        bias_4: Input = Input(shape=(4,4,1),  dtype='float32', name='bias_4')
    else:
        bias_26: Input = Input(shape=(16,16,1),  dtype='float32', name='bias_26')
        bias_24: Input = Input(shape=(16,16,1),  dtype='float32', name='bias_24')
        bias_12: Input = Input(shape=(8,8,1),  dtype='float32', name='bias_12')
        bias_6: Input = Input(shape=(4,4,1),  dtype='float32', name='bias_6')
        bias_4: Input = Input(shape=(4,4,1),  dtype='float32', name='bias_4')

    inputs: List[Input] =[input_image, input_matrix, hidden_state_1, cell_state_1, hidden_state_2, cell_state_2, bias_26, bias_24, bias_12, bias_6, bias_4]

    ########################
    ###  Connect Layers  ###
    ########################
    T:TypeVar = TypeVar('T',BilinearInterpolation,Conv2D,add,Any)
    ## Glimpse Network
    # spatial transformer, performs affine transformation of input image to a 26x26 patch
    # x:T= BilinearInterpolation(glimpse_size, clip_value)([input_image, input_matrix])
    x:BilinearInterpolation = BilinearInterpolation(height=glimpse_size[0], width=glimpse_size[1])
    print('INPUT IMAGE:',input_image.shape)
    x:Tensor = x([input_image, input_matrix]);
    print("Bilinear Interpolation from tEDRAM cell:", x.shape) # ? x 26 x 26 x ?
    bn_axis: int = 3

    if(unique_glimpse!=0):
        x = Conv2D(int(n_filters/2), (5,5), padding='same', activation='relu', use_bias=False, name='glimpse_conv1')(x)
        b: LocallyConnected2D = LocallyConnected2D(int(n_filters/2), (1,1), padding='valid', use_bias=False, name='glimpse_conv1_bias')(bias_26)
    else:
        x = conv_1(x)
        b = conv_1_bias(bias_26)
    x = add([x, b], name='glimpse_conv1_add')
    if bn: x = BatchNormalization(axis=bn_axis, name='glimpse_conv1_bn')(x)
    x = conv_2(x)
    b = conv_2_bias(bias_24)
    x = add([x, b], name='glimpse_conv2_add')
    if bn: x = BatchNormalization(axis=bn_axis, name='glimpse_conv2_bn')(x)
    x = max_pooling_1(x)
    x = conv_3(x)
    b = conv_3_bias(bias_12)
    x = add([x, b], name='glimpse_conv3_add')
    if bn: x = BatchNormalization(axis=bn_axis, name='glimpse_conv3_bn')(x)
    x = conv_4(x)
    b = conv_4_bias(bias_12)
    x = add([x, b], name='glimpse_conv4_add')
    if bn: x = BatchNormalization(axis=bn_axis, name='glimpse_conv4_bn')(x)
    x = max_pooling_2(x)
    x = conv_5(x)
    b = conv_5_bias(bias_6)
    x = add([x, b], name='glimpse_conv5_add')
    if bn: x = BatchNormalization(axis=bn_axis, name='glimpse_conv5_bn')(x)
    x = conv_6(x)
    b = conv_6_bias(bias_4)
    x = add([x, b], name='glimpse_conv6_add')
    if bn: x = BatchNormalization(axis=bn_axis, name='glimpse_conv6_bn')(x)
    x = flatten(x)
    x_what = glimpse_what(x)
    if bn: x_what = BatchNormalization(name='glimpse_dense_bn')(x_what)
    x_where = glimpse_where(input_matrix)
    if bn: x_where = BatchNormalization(name='glimpse_localisation_bn')(x_where)
    x = multiply([x_where, x_what], name='glimpse_output')

    ## RNNs
    x = reshape_to_sequence(x);
    rnn1:ndarray = None;
    h1:ndarray = None;
    c1:ndarray=None;
    rnn2:ndarray=None;
    h2:ndarray=None;
    c2:ndarray=None;
    rnn1, h1, c1 = LSTM_classify(x, initial_state=[hidden_state_1, cell_state_1]) #type: ndarray, ndarray, ndarray
    if output_localisation:
        rnn2, h2, c2 = LSTM_localize(rnn1, initial_state=[hidden_state_2, cell_state_2]) #type: ndarray, ndarray, ndarray
    rnn1= reshape_from_sequence(rnn1);

    # apply dropout
    if do:
        rnn1 = Dropout(dropout, name='classification_dropout')(rnn1)
        if output_localisation:
            rnn2 = Dropout(dropout, name='localisation_dropout')(rnn2)


    ## Classification Network
    x = cla_fc_1(rnn1)
    if bn: x = BatchNormalization(name='classification_fc1_bn')(x)
    x = cla_fc_2(x)
    if bn: x = BatchNormalization(name='classification_fc2_bn')(x)
    classification = cla_fc_3(x)

    ## Emission Network - outputs the flat localization matrix
    localisation:Dense=None
    if(output_localisation):
        if (unique_emission):
            localisation = Dense(6, activation='tanh', weights=emission_weights(RNN_size_2, emission_bias), name='emission')(rnn2)
        else:
            localisation = em(rnn2)


    ########################
    ###  Define Outputs  ###
    ########################
    if output_localisation:
        outputs=[classification, localisation, h1, c1, h2, c2]
    else:
        outputs=[classification, h1, c1]

    return Model(inputs, outputs, name='edram_cell_'+str(step))
