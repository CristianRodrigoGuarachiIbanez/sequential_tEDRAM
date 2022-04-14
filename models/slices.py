
from tensorflow.keras.layers import Layer
from tensorflow import slice, squeeze
from tensorflow import Tensor
from typing import Dict, List

class Slice(Layer):
    def __init__(self, begin:List[int], size: List[int],**kwargs) -> None:
        super(Slice, self).__init__(**kwargs)
        self.begin: List[int] = begin
        self.size: List[int] = size
    def get_config(self) -> Dict[str, int]:

        config = super().get_config().copy()
        config.update({
            'begin': self.begin,
            'size': self.size,
        })
        return config
    def call(self, inputs: Tensor) -> Tensor:
        x: Tensor = slice(inputs, self.begin, self.size)
        return squeeze(x, 1)

if __name__ == '__main__':
   pass
