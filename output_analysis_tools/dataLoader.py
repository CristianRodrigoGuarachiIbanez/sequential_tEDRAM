from __future__ import print_function

from h5py import File
from pickle import load
from tensorflow.keras.models import load_model, Model
from h5py import File
from json import loads, dumps
from numpy import ndarray, zeros, ones, array, asarray, vstack, argmax, dot, mean as Mean, square, std as Std, var, empty
from config import config, datasets

from typing import List, Dict, Tuple, Sized, Union, TypeVar

B: TypeVar = TypeVar("B", Tuple, str)
D: TypeVar = TypeVar("D", File, ndarray)
class DataLoader:
    # default training parameters
    _batch_size = 192
    _model_id = 2
    _n_steps = 6
    # paths to the datasets
    _datasets:List[str];

    collision_labels = ["collision_hand", "collision_forearm", "collision_both", "collision_none"]
    def __init__(self)->None:
        self._n_train:int=0;
        self._n_test:int=0;
        self._features: ndarray = None;
        self._labels: List[List[int]] = None;
        self._locations: ndarray = None;
        self._file=None;
        self._datasets = datasets;
    def get_test(self, test:bool=False)->int:
        if(test==True):
            return self._n_test
        else:
            return self._n_train
    def get_images(self, labels:bool=False)->ndarray:
        if(labels is True):
            return array(self._labels)
        else:
            return self._features
    def set_n_test(self, values:int)->None:
        self._n_test = values;
    def set_n_train(self, values)->None:
        self._n_train = values
    def set_features(self,img:ndarray)->None:
        self._features = img
    def set_labels(self,labels:List[List[int]])->None:
        self._labels = labels
    def _set_file(self, filename:str)->File:
        return File(filename, 'r+')
    def _recov_data(self, dataset_id:int, img:bool=True)->D:

        data_path:str = self._datasets[dataset_id+1]
        print("\n[Info] Opening features data", data_path)
        labels_path:str = self._datasets[1];
        print("\n[Info] Opening labels data", labels_path)

        # ---------------------- load the image data -----------------------
        if(img is True):
            try:
                # print('[Info] image data was successfully opened!')
                return File(data_path, 'r')
            except Exception:
                print("[Error]", data_path, "does not exist.\n")
                exit()
        # --------------------- load label data --------------------
        else:
            try:
                file = open(labels_path, 'rb')
                # print('[Info] labels data " {} "  was successfully opened'.format(len(labels)));
                labels = load(file);
                file.close();
                return labels
            except Exception as e:
                print("[Error] troubles at loading the daata from pickle:", e);
                exit()
    @staticmethod
    def recov_path(dataset_id:int)->B:
        # mode = 0 if output_init_matrix==0 and mode==0 else 1
        if (dataset_id < 2):
            return config['input_shape_scene']
        elif (dataset_id == 2):
            return config['input_shape_binocular']
        else:
            return config['input_shape_scene']

    def fix_layer0(self, filename:str, batch_input_shape:List[int], dtype:str):
        self._file = self._set_file(filename)
        model_config = loads(self._file.attrs['model_config'].decode('utf-8'))
        layer0 = model_config['config']['layers'][0]['config']
        layer0['batch_input_shape'] = batch_input_shape
        layer0['dtype'] = dtype
        self._file.attrs['model_config'] = dumps(model_config).encode('utf-8')

    def init_dataset(self, dataset_id:int):
        # ---------------------- load the image data -----------------------
        data:D = self._recov_data(dataset_id, img=True);
        # --------------------- load label data --------------------
        labels: ndarray = self._recov_data(dataset_id, img=False)
        if (dataset_id==1):
            self.set_n_train(data['feature_data']['scene_data'].shape[0] - 22671);
            self.set_n_test(data['feature_data']['scene_data'].shape[0] - self._n_train)  # hier ist 22671

            if((self._n_train >0) and (self._n_test >0)):
                self.set_features(data['feature_data']['scene_data'][:self._n_train])
                self.set_labels(labels[:self._n_train]) # --- 0. left_hand, 1. right_hand, 2.left_forearm, 3. right_forearm
            else:
                print('the size of n_train and n_test was not set!')

        elif(dataset_id==2):
            self.set_n_train(data['feature_data']['binocular_data'].shape[0] - 22671)
            self.set_n_test(data['feature_data']['binocular_data'].shape[0] - self._n_train ) # hier ist 22671
            if((self._n_train >0) and (self._n_test >0)):
                self.set_features( data['feature_data']['binocular_data'][:self._n_train])
                self.set_labels(labels[:self._n_train]) # --- 0. left_hand, 1. right_hand, 2.left_forearm, 3. right_forearm
            else:
                print('the size of n_train and n_test was not set!')
        else:
            print('[Dataset ID] dataset_id is not 1')