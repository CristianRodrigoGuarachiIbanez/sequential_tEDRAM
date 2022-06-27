from h5py import File
from pickle import load, dump
from numpy import ndarray, asarray, uint8, float32
from os import getcwd
from cython_modules.disparity_maps_cython.sampler.ImgSampler import CSampler
from cython_modules.disparity_maps_cython.sum_disparity_map.sum_disparity_arrays.sumDisparityArrays import SumDisparityArrays
from cython_modules.disparity_maps_cython.merge_dimensions.merge_4dim.mergeDisparityArrays import MergeDisparityArrays
from cython_modules.disparity_maps_cython.merge_dimensions.merge_5dim.mergeDim import MergeArrayDimensions
from dataset_tools.h5Writer import H5Writer
from typing import List
from cv2 import imwrite
import cython
class Sampler(object):
    _file:File;
    _writer:dump
    _reader:load;
    _data:ndarray;
    _sample:CSampler
    def __init__(self, filename:str)->None:
        self.__file(filename)
        self._writer = None;
    def get_sample(self, samp=1000)->ndarray:
        self.__sampling(samp)
        return self._sample
    def __file(self, filename):
        try:
            #print(getcwd())
            self._file = File(getcwd() + filename, "r")
            print(self._file.keys())
        except Exception as e:
            raise Exception(e)
    def recover_dataset(self, datasetname:str)->None:
        """
        "disparity_arrays_56"
        :param datasetname:
        :return:
        """
        self._data = self._file[datasetname]
        print("SHAPE:", self._data.shape)
    def __sampling(self, samp:int)->None:
        #sampler = CSampler(asarray(self._data,dtype=uint8), samp)
        #self._sample = sampler.get_img()
        if(self._data.ndim==3):
            self._sample = self._data[:samp,:,:]
        elif (self._data.ndim == 4):
            self._sample = self._data[samp:samp*2, :, :, :]
        else:
            self._sample = self._data[:samp,:,:,:,:]
    def _sum_img(self)->None:
        #self.recover_dataset()
        self._sample = SumDisparityArrays(asarray(self._data, dtype=float32))
        print("SHAPE SUM of ARRAYS:", self._sample.shape)
    def _merge_dim(self) ->None:
        #data = asarray(self._data, dtype=float32)
        self._sample = list()
        i:cython.int
        for i in range(self._data.shape[0]):
            img = MergeDisparityArrays(self._data[i,:,:,:,:])
            vector = img.get_vector()
            print("index;", i, "shape:", vector.shape)
            self._sample.append(vector)
    def _merge_arrays(self):
        self._sample = MergeArrayDimensions(self._data)
    def write_data(self, out_file:str,samp:int)->None:
        sample: ndarray
        directory:str = getcwd()
        self.__sampling(samp)
        with open(directory +"/"+ out_file, "wb") as file:
            dump(self._sample,file)
            file.close()
        print("sample was successfully saved!")

    def write_h5data(self, out_file:str, dataset:List[str]) ->None:
        if(out_file == "/training_data/disparity_maps_56.h5"):
            self._merge_dim()
            #self._merge_arrays()
            self._sample = asarray(self._sample, dtype=float32)
        elif(out_file == "/training_data/sum_of_disparity_maps.h5"):
            self._sum_img()
            self._sample = asarray(self._sample, dtype=float32)
        else:
            raise Exception("not valid name")
        file = H5Writer(getcwd() + out_file);
        if(self._sample is not None) and (len(self._sample.shape)==4):
            file.saveImgDataIntoGroup([self._sample], " ", dataset)

if(__name__=="__main__"):
    file:str = r"/training_data/disparity_maps_56imgs.h5"#AffectNet_train_data_keras.h5" #AffectNet_train_data_keras.h5
    output = r"/scratch/gucr/tEDRAM2/training_data/"

    datasetname:str ="disparity_arrays_56" # "disparity_arrays_s" #'emotions'
    s = Sampler(filename=file)
    s.recover_dataset(datasetname=datasetname)
    s.write_data("sample_disparity_maps.txt", 2000)
    # sample = s.get_sample()
    #s.write_h5data(output, ['disparity_array_56'])
    #sample = s.get_sample()
    #print("sample shape ->", sample.shape)
    #for i in range(len(sample)):
        #imwrite(output +"scene_"+str(i)+".png", sample[i,:,:])
    #print(sample.shape)

    
