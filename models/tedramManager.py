from typing import List, Tuple
from .tedram_model import tedram_model
from .tedram_model_56img import tedram_model_56imgs
from .tedram_one_output_model import tedram_op_model
from tensorflow.keras.models import Model
class tEDRAM_TF:
    input_shape:Tuple
    model:Model
    def __init__(self, image_shape:Tuple)->None:
        self.input_shape = image_shape
        self.model = None
    def get_model(self)->Model:
        return self.model
    def set_model(self, model:Model)->None:
        self.model = model
    def create_tedram_model(self,
                               learning_rate: float = 0.0001, n_steps: int = 10, glimpse_size: Tuple[int, int] = (26, 26), coarse_size: Tuple[int, int] = (12, 12),
                               n_filters: int = 128, filter_sizes: Tuple[int, int] = (3, 5), n_features: int = 1024, RNN_size_1: int = 512, RNN_size_2: int = 512,
                               n_classes: int = 6, output_mode: int = 0, use_init_matrix: int = True, emission_bias: int = 1, clip_value: int = 1, unique_emission: int = False,
                               unique_glimpse: int = False, bn: bool = True, dropout: int = 0, use_weighted_loss: int = False, localisation_cost_factor: float = 1.0)->None:

        if(self.input_shape[1]==120):
            self.model = tedram_model(input_shape=self.input_shape, learning_rate=learning_rate, steps=n_steps, glimpse_size=glimpse_size, coarse_size=coarse_size, n_filters=n_filters,
                         filter_sizes=filter_sizes, n_features=n_features,  RNN_size_1=RNN_size_1, RNN_size_2=RNN_size_2, n_classes=n_classes,
                         output_mode=output_mode, use_init_matrix=use_init_matrix, emission_bias=emission_bias, clip_value=clip_value,
                         unique_emission=unique_emission, unique_glimpse=unique_glimpse,  bn=bn, dropout=dropout,
                         use_weighted_loss=use_weighted_loss, localisation_cost_factor=localisation_cost_factor);
        elif(self.input_shape[1]==56):
            self.model = tedram_model_56imgs(input_shape=self.input_shape, learning_rate=learning_rate, steps=n_steps, glimpse_size=glimpse_size, coarse_size=coarse_size, n_filters=n_filters,
                         filter_sizes=filter_sizes, n_features=n_features,  RNN_size_1=RNN_size_1, RNN_size_2=RNN_size_2, n_classes=n_classes, output_mode=output_mode,
                         use_init_matrix=use_init_matrix, emission_bias=emission_bias, clip_value=clip_value, unique_emission=unique_emission, unique_glimpse=unique_glimpse,
                         bn=bn, dropout=dropout, use_weighted_loss=use_weighted_loss, localisation_cost_factor=localisation_cost_factor)

        elif (self.input_shape[1] == 7):
            self.model = tedram_model_56imgs(input_shape=self.input_shape, learning_rate=learning_rate, steps=n_steps, glimpse_size=glimpse_size, coarse_size=coarse_size,
                                             n_filters=n_filters, filter_sizes=filter_sizes, n_features=n_features, RNN_size_1=RNN_size_1, RNN_size_2=RNN_size_2, n_classes=n_classes,
                                             output_mode=output_mode, use_init_matrix=use_init_matrix, emission_bias=emission_bias, clip_value=clip_value, unique_emission=unique_emission,
                                             unique_glimpse=unique_glimpse, bn=bn, dropout=dropout, use_weighted_loss=use_weighted_loss, localisation_cost_factor=localisation_cost_factor)

        else:
            self.model = tedram_model(input_shape=self.input_shape, learning_rate=learning_rate, steps=n_steps, glimpse_size=glimpse_size, coarse_size=coarse_size, n_filters=n_filters,
                                      filter_sizes=filter_sizes, n_features=n_features, RNN_size_1=RNN_size_1, RNN_size_2=RNN_size_2, n_classes=n_classes,
                                      output_mode=output_mode, use_init_matrix=use_init_matrix, emission_bias=emission_bias, clip_value=clip_value,
                                      unique_emission=unique_emission, unique_glimpse=unique_glimpse, bn=bn, dropout=dropout, use_weighted_loss=use_weighted_loss,
                                      localisation_cost_factor=localisation_cost_factor);

    def create_one_output_model(self, learning_rate: float = 0.0001, n_steps: int = 10, glimpse_size: Tuple[int, int] = (26, 26), coarse_size: Tuple[int, int] = (12, 12),
                               n_filters: int = 128, filter_sizes: Tuple[int, int] = (3, 5), n_features: int = 1024, RNN_size_1: int = 512, RNN_size_2: int = 512,
                               n_classes: int = 6, output_mode: int = 0, use_init_matrix: int = True, emission_bias: int = 1, clip_value: int = 1, unique_emission: int = False,
                               unique_glimpse: int = False, bn: bool = True, dropout: int = 0, use_weighted_loss: int = False, localisation_cost_factor: float = 1.0):

        self.model = tedram_op_model(input_shape=self.input_shape, learning_rate=learning_rate, steps=n_steps, glimpse_size=glimpse_size, coarse_size=coarse_size, n_filters=n_filters,
                                      filter_sizes=filter_sizes, n_features=n_features, RNN_size_1=RNN_size_1, RNN_size_2=RNN_size_2, n_classes=n_classes, output_mode=output_mode, use_init_matrix=use_init_matrix, emission_bias=emission_bias, clip_value=clip_value,
                                      unique_emission=unique_emission, unique_glimpse=unique_glimpse, bn=bn, dropout=dropout, use_weighted_loss=use_weighted_loss,
                                      localisation_cost_factor=localisation_cost_factor)