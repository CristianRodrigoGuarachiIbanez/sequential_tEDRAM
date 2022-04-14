from tensorflow.keras import backend as K
from tensorflow import (shape, range, reshape, gather_nd, tile, stack, linspace, meshgrid, expand_dims, cast, ones_like,
float32, int32, matmul, transpose, name_scope, math, clip_by_value, Tensor, concat, repeat, TensorShape )
from tensorflow.keras.layers import Layer
from typing import List, Tuple, Dict, Any
from tensorflow import Tensor
from tensorflow.keras.layers import Input
if K.backend() == 'tensorflow':
    import tensorflow as tf


    def K_meshgrid(x, y):
        return tf.meshgrid(x, y)


    def K_linspace(start, stop, num):
        return tf.linspace(start, stop, num)

else:
    raise Exception("Only 'tensorflow' is supported as backend")


class BilinearInterpolation(Layer):
    """Performs bilinear interpolation as a keras layer
    References
    ----------
    [1]  Spatial Transformer Networks, Max Jaderberg, et al.
    [2]  https://github.com/skaae/transformer_network
    [3]  https://github.com/EderSantana/seya
    """
    def __init__(self, output_size:Tuple[int,int], clip_max:int, **kwargs):
        self.output_size = output_size
        self.clip_max = clip_max
        super(BilinearInterpolation, self).__init__(**kwargs)

    def __call__(self, tensors:List[Input]):
        return self.call(tensors)

    def get_config(self):
        return {
            'output_size': self.output_size,
        }
    def compute_output_shape(self, input_shapes):
        height, width = self.output_size
        num_channels = input_shapes[0][-1]
        return (None, height, width, num_channels)

    def call(self, tensors:List[Input]):
        X, t = tensors # type:Input,Input
        output:Tensor;
        if(self.clip_max < 1):
            # clip zoom values of transformation matrix
            t_clipped = stack((clip_by_value(t[:, 0], .0, self.clip_max), t[:, 1], t[:, 2], t[:, 3],
                                 clip_by_value(t[:, 4], .0, self.clip_max), t[:, 5]), axis=1)
            output = self._transform(X, t_clipped, self.output_size)
        else:
            output = self._transform(X, t, self.output_size)
        return output

    def _interpolate(self, image: Input, sampled_grids, output_size: Tuple[int, int]):

        batch_size = shape(image)[0]  # 10
        height = shape(image)[1]  # 120
        width = shape(image)[2]  # 160
        num_channels = shape(image)[3]  # 1

        x = cast(reshape(sampled_grids[:, 0:1, :],(-1,)), dtype='float32')  # sample_grid: (None, 3, 676) -> nur index 0 der zweiten Dim -> None, 1, 676
        y = cast(reshape(sampled_grids[:, 1:2, :],(-1,)), dtype='float32')  # nur index 1 der zweiten Dim -> None, 1, 676 ->  None * 676
        # Werte innerhalb des  Tensors werden geändert
        x = .5 * (x + 1.0) * cast(height, dtype='float32')
        y = .5 * (y + 1.0) * cast(width, dtype='float32')
        # Werte innerhalb des Tensor  von Float zu Int
        x0 = cast(x, 'int32')
        x1 = x0 + 1  # werte des Tensor + 1 geändert
        y0 = cast(y, 'int32')
        y1 = y0 + 1

        max_x = int(image.shape[1]-1)#int(K.int_shape(image)[1] - 1)  # int_shape:  (None, 120, 160,1)[1] -1 -> 119
        max_y = int(image.shape[2]-1)#int(K.int_shape(image)[2] - 1)  # -> 159

        x0 = clip_by_value(x0, 0, max_x)  # tensor mit Clipping Werte  0, 119
        x1 = clip_by_value(x1, 0, max_x)  # tensor mit clipping Werte 0, 119
        y0 = clip_by_value(y0, 0, max_y)  # 0, 159
        y1 = clip_by_value(y1, 0, max_y)
        # ------ create the indices for the pixels
        pixels_batch = range(0, batch_size) * ( height * width)  # *(height * width) definiert die Step mit dem die Array Werte zunehmend ansteigen z.B. [0, 19200, 38400, 57600, ..], wenn  *(height * width) = 100
        pixels_batch = expand_dims(pixels_batch, axis=-1)  # shape vor : (batch_size, ) -> (batch_size, 1) = (10, 1)
        flat_output_size = output_size[0] * output_size[1]  # 26 *26 =676
        base = repeat(pixels_batch, flat_output_size, axis=1)  # pixels_batch = (batch_size, 1) -> (batch_size, 1*flat_output_size) -> (10, 676)
        base = reshape(base,(-1,))  # 10 * 676 = 6760

        # base_y0 = base + (y0 * width)
        base_y0 = y0 * width  # (0,159) * 160
        base_y0 = base + base_y0
        # base_y1 = base + (y1 * width)
        base_y1 = y1 * width
        base_y1 = base_y1 + base

        indices_a = base_y0 + x0  # shape=(?,)
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        print('values indices a:{} , values indices b: {}'.format(indices_a, indices_b))  # shape (None,)
        flat_image = reshape(image, shape=(-1, num_channels))  # [[0.],[0.],..]  -> shape=(?, ?), length: 192000
        print('flat image -> Bilinear Class: {}'.format(flat_image))
        flat_image = cast(flat_image, dtype='float32')
        # print('flat image casted -> Bilinear Class: {}'.format(flat_image))

        pixel_values_a = gather_nd(flat_image, indices_a)  # (?, ?)
        # tensor = K.eval(pixel_values_a)
        print('pixel_values -> from interpolation:  {}'.format(K.print_tensor(pixel_values_a)))
        pixel_values_b = gather_nd(flat_image, indices_b)
        # print('pixel_values -> from interpolation:  {}'.format(pixel_values_b))
        pixel_values_c = gather_nd(flat_image, indices_c)
        pixel_values_d = gather_nd(flat_image, indices_d)

        x0 = cast(x0, 'float32')
        # print('values x0: {}'.format(x0))
        x1 = cast(x1, 'float32')
        y0 = cast(y0, 'float32')
        y1 = cast(y1, 'float32')

        area_a = expand_dims(((x1 - x) * (y1 - y)), 1)  # shape=(?, 1),
        # print('area_a -> bilinear class: {}'.format(area_a))
        area_b = expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = expand_dims(((x - x0) * (y - y0)), 1)

        values_a = area_a * pixel_values_a
        # print('values a -> bilinear class: {}'.format(K.eval(values_a))
        values_b = area_b * pixel_values_b
        values_c = area_c * pixel_values_c
        values_d = area_d * pixel_values_d
        return values_a + values_b + values_c + values_d

    def _make_regular_grids(self, batch_size, height, width):
        # making a single regular grid
        x_linspace:Tensor = K_linspace(-1., 1., width)
        y_linspace:Tensor = K_linspace(-1., 1., height)
        x_coordinates, y_coordinates = K_meshgrid(x_linspace, y_linspace)
        x_coordinates = reshape(x_coordinates,(-1,))
        y_coordinates = reshape(y_coordinates,(-1,))
        ones = ones_like(x_coordinates)
        grid = concat([x_coordinates, y_coordinates, ones], 0)

        # repeating grids for each batch
        grid = reshape(grid,(-1,))
        grids = tile(grid, stack([batch_size]))
        return reshape(grids, (batch_size, 3, height * width))

    def _transform(self, X:Input, affine_transformation:Input, output_size:Tuple[int, int]):
        print('INPUT:{} and {}'.format(X.shape,X.get_shape()))
        shapei:Tensor = X.get_shape()
        batch_size, num_channels = self.__convert_to_tensor(shapei[0]), self.__convert_to_tensor(shapei[3]) #shape(X)[0], shape(X)[3]
        print('BATCH SIZE FROM BINOCULAR',batch_size, 'NUMBER OF CHANNELS',num_channels)
        transformations = reshape(affine_transformation, shape=(batch_size, 2, 3))  # ? x  2 x 3
        # transformations = K.cast(affine_transformation[:, 0:2, :], 'float32')
        regular_grids = self._make_regular_grids(batch_size, *output_size)  # bilinear_interpolation_i/Reshape_4:0", shape=(?, 3, 676), dtype=float32
        print('REGULAR GRIDS:{}'.format(regular_grids))
        sampled_grids = K.batch_dot(transformations, regular_grids)
        print('SAMPLED_GRIDS:{}'.format(sampled_grids))
        interpolated_image = self._interpolate(X, sampled_grids, output_size)  # (None, 10,120,160,1) |  (None, 3, 676) | (26, 26)
        # print('Reshape from Binocular:{}'.format(interpolated_image))
        new_shape = (batch_size, output_size[0], output_size[1], num_channels)
        interpolated_image = reshape(interpolated_image, new_shape)
        #print('Reshape from Binocular:{}'.format(interpolated_image))
        return interpolated_image
    @staticmethod
    def __convert_to_tensor(elem:Tuple[int])->Tensor:
        #elem = tf.convert_to_tensor(elem , dtype=int32)
        elem = TensorShape(elem)
        return elem
