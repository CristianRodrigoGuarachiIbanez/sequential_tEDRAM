from tensorflow.keras.layers import Layer #tf.keras.layers.
from tensorflow import (shape, range, reshape, gather_nd, tile, stack, linspace, meshgrid, expand_dims, cast, ones_like,
float32, int32, matmul, transpose, name_scope, math, clip_by_value, Tensor )
from tensorflow.keras.layers import Input
from typing import List, Tuple, Any, TypeVar, Dict, Union

class BilinearInterpolation(Layer):
    def __init__(self, height:int =40, width:int=40):
        super(BilinearInterpolation, self).__init__()
        self.height:int = height
        self.width:int = width

    def compute_output_shape(self, input_shape) -> List[Union[None, int]]:
        return [None, self.height, self.width, 1]

    def get_config(self) -> Dict[str, int]:
        return {
            'height': self.height,
            'width': self.width,
        }

    def build(self, input_shape) -> None:
        print("Building Bilinear Interpolation Layer with input shape:", input_shape)

    def advance_indexing(self, inputs: Input, x:Tensor, y:Tensor) -> Tensor:
        '''
        Numpy like advance indexing is not supported in tensorflow, hence, this function is a hack around the same method
        '''
        shapei: Tensor = shape(inputs)
        batch_size, _, _ = shapei[0], shapei[1], shapei[2] #type: Tensor

        batch_idx: Tensor = range(0, batch_size)
        batch_idx: Tensor = reshape(batch_idx, (batch_size, 1, 1))
        b: Tensor = tile(batch_idx, (1, self.height, self.width))
        indices = stack([b, y, x], 3)
        return gather_nd(inputs, indices)

    def call(self, inputs: List[Input]) -> Tensor:
        images, theta = inputs #type: Input, Input
        homogenous_coordinates = self.grid_generator(batch=shape(images)[0])
        return self.interpolate(images, homogenous_coordinates, theta)

    def grid_generator(self, batch: int) -> Tensor:
        x: Tensor = linspace(-1.0, 1.0, self.width)
        y: Tensor = linspace(-1.0, 1.0, self.height)

        xx, yy = meshgrid(x, y)  # type: Tensor, Tensor;
        xx = reshape(xx, (-1,))
        yy = reshape(yy, (-1,))
        print('xx values: {}, yy values: {}'.format(xx, yy))
        homogenous_coordinates: Tensor = stack([xx, yy, ones_like(xx)])
        #print(homogenous_coordinates)
        homogenous_coordinates = expand_dims(homogenous_coordinates, axis=0)
        #print(homogenous_coordinates)
        homogenous_coordinates = tile(homogenous_coordinates, [batch, 1, 1])
        #print(homogenous_coordinates)
        homogenous_coordinates = cast(homogenous_coordinates, dtype=float32)
        return homogenous_coordinates
    def interpolate(self, images:Input, homogenous_coordinates:Tensor, theta:Input) -> Tensor:
        transformed:Tensor
        x_transformed:float
        y_transformed:float
        x:Tuple[Tensor]
        y:Tuple[Tensor]
        x0:int
        x1:int
        y0:int
        y1:int
        with name_scope("Transformation"):
            print('theta: {}, coodinates: {}'.format(theta, homogenous_coordinates))
            transformed = matmul(theta, homogenous_coordinates);
            transformed = transpose(transformed, perm=[0, 2, 1]);
            transformed = reshape(transformed, [-1, self.height, self.width, 2]);

            x_transformed = transformed[:, :, :, 0]
            y_transformed = transformed[:, :, :, 1]

            x = ((x_transformed + 1.) * cast(self.width, dtype=float32)) * 0.5
            y = ((y_transformed + 1.) * cast(self.height, dtype=float32)) * 0.5

        with name_scope("VariableCasting"):
            x0 = cast(math.floor(x), dtype=int32)
            x1 = x0 + 1
            y0 = cast(math.floor(y), dtype=int32)
            y1 = y0 + 1

            x0:Tensor = clip_by_value(x0, 0, self.width - 1)
            x1:Tensor  = clip_by_value(x1, 0, self.width - 1)
            y0:Tensor = clip_by_value(y0, 0, self.height - 1)
            y1:Tensor = clip_by_value(y1, 0, self.height - 1)
            x:Tensor = clip_by_value(x, 0,cast(self.width, dtype=float32) - 1.0)
            y:Tensor = clip_by_value(y, 0, cast(self.height, dtype=float32) - 1)
        Ia: Tensor
        Ib: Tensor
        Ic: Tensor
        Id: Tensor
        with name_scope("AdvanceIndexing"):
            Ia = self.advance_indexing(images, x0, y0)
            Ib = self.advance_indexing(images, x0, y1)
            Ic = self.advance_indexing(images, x1, y0)
            Id = self.advance_indexing(images, x1, y1)

        wa:int
        wb:int
        wc:int
        wd:int
        x:int
        y:int
        with name_scope("Interpolation"):
            x0 = cast(x0, dtype=float32)
            x1 = cast(x1, dtype=float32)
            y0 = cast(y0, dtype=float32)
            y1 = cast(y1, dtype=float32)

            wa = (x1 - x) * (y1 - y)
            wb = (x1 - x) * (y - y0)
            wc = (x - x0) * (y1 - y)
            wd = (x - x0) * (y - y0)

            wa = expand_dims(wa, axis=3)
            wb = expand_dims(wb, axis=3)
            wc = expand_dims(wc, axis=3)
            wd = expand_dims(wd, axis=3)

        return math.add_n([wa * Ia + wb * Ib + wc * Ic + wd * Id])


