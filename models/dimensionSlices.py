from tensorflow import Tensor
from typing import Dict, List
from .slices import Slice
from .dim_AutoReducer.autoReducer import ReduceDimensions
class DimensionSlices(object):
    data:Tensor
    input_size:Tensor
    def __init__(self, inputs:Tensor) -> None:
        #super(Slice, self).__init__(**kwargs)
        self.data = inputs

    def get_input_size(self) ->Tensor:
        return self.input_size

    def slices(self, start:int, goal:int, indices:List[int])->None:
        print("START:", start)
        reducer: ReduceDimensions = ReduceDimensions(size=start)
        nulls = reducer.get_nulls(0, indices[0]).tolist()
        ones = reducer.get_ones().tolist()
        slicer = Slice(nulls, ones)
        self.input_size = slicer(self.data)
        print("reduced input shape:", self.input_size.shape)
        i:int =0;
        while (len(self.input_size.shape)>goal):
            start -=1
            slicer = Slice(reducer.get_nulls(start, indices[i]).tolist(), reducer.get_ones(start).tolist())
            self.input_size = slicer(self.input_size)
            i+=1
            print("reduced input shape:",self.input_size.shape)