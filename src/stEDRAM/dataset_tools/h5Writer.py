#import cv2 as cv
from typing import List, Dict, Tuple, TypeVar, Any, Callable, Union, KeysView, AbstractSet, Generator
import logging
import sys
import cython
from numpy import ndarray, asarray, uint8
from h5py import File, Group, Dataset
from numpy import ndarray

# ----------------------------------------- CAMERAS COORDINATES WRITER -----------------------------------------------
class H5Writer:
    L = TypeVar("L", List[ndarray], Dict[str, ndarray], ndarray)
    #__H5PY: h5py.File = h5py.File(fileobj=None, mode=None)
    def __init__(self, filename: str) -> None:
        self.__file = File(filename, 'a')

    def saveImgDataIntoGroup(self,  imgData: L, groupName: str, datasetNames: List[str]) -> None:
        #with File(filename, 'a') as file:
        assert (len(imgData) == len(datasetNames)), 'the number of data to save and data set names are no equal'
        if (len(imgData) > 1 and len(datasetNames) > 1):
            group: Group = self.__file.create_group(groupName)
            print('... group was created successfully!')
            for i in range(len(datasetNames)):
                group.create_dataset(datasetNames[i], data=asarray(imgData[i]), compression='gzip', compression_opts=9)
                print('... dataset was created successfully!')
        else:
            self.__file.create_dataset(datasetNames[0], data=asarray(imgData[0], dtype=uint8), compression='gzip', compression_opts=9)
            print('... dataset was created successfully!')
    def saveDataIntoGroups(self, data:List, group:List, dataset:List[List]):
        i:cython.int
        j:cython.int
        group:Group
        assert (type(group)==list), "the group should be a list of group names"
        for i in range(len(group)):
            group_temp = self.__file.create_group(str(group[i]));
            print('... group was created successfully!')
            for j in range(len(dataset[i])):
                group_temp.create_dataset(dataset[i][j], data=asarray(data[i][j], dtype='float32'), compression='gzip', compression_opts=9)
                print('... dataset was created successfully!')
        print( '... data was saved successfully!')
        self.closingH5PY()
    def loadImgDataFromGroup(self, groupName: str = None, datasetNames: str = None) -> Generator:
        # with File(filename, "r") as file:
        keys: List[str] = list(self.__file.keys())
        imgArray: ndarray = None;
        if (keys):
            print(keys);
        else:
            pass
        if (groupName):
            group: Group = self.__file.get(groupName)  # group2 = hf.get('group2/subfolder')
            items: List[Tuple] = list(group.items()) if group is not None else print('group could not be recovered')  # [(u'data3', <HDF5 dataset "data3": shape (100, 3333), type "<f8">)]
            if (len(items) >0):
                print(items)
                try:
                    for i in range(len(items)):
                        print('recovering group class:',group.get(items[i][0]))
                        yield asarray(group.get(items[i][0]))  # n1 = group1.get('data1') \n np.array(n1).shape
                except StopIteration:
                     self.closingH5PY()
            else: pass
        elif (datasetNames):
            #cls.__closingH5PY(file)
            yield self.__file.get(datasetNames)  # n1 = group1.get('data1') \n np.array(n1).shape

            #print("Size of List", len(imgArray), "Size of Tuple", len(imgArray[0]), "Size of Array", imgArray[0][0].shape, imgArray[0][0].size)

    def closingH5PY(self) -> None:
        self.__file.close()

class H5GTWriter:
    L = TypeVar("L", List[ndarray], Dict[str, ndarray])
    @classmethod
    def saveGTDataIntoGroup(cls,  filename: str, GTData: L, groupName: str, datasetNames: List[str]) -> None:
        with File(filename, 'a') as file:
            group: Group = file.create_group(groupName)
            print('... group was created successfully!')
            assert (len(GTData) == len(datasetNames)), 'the number of data to save and data set names are no equal'
            for i in range(len(datasetNames)):
                group.create_dataset(datasetNames[i], data=asarray(GTData[i]), compression='gzip', compression_opts=9)
                print('... dataset was created successfully!')

    @classmethod
    def loadGTDataFromGroup(cls, filename: str, groupName: str = None, datasetNames: str = None) -> None:
        with File(filename, "r") as file:
            keys: List[str] = list(file.keys())
            imgArray: ndarray = None;
            if (keys):
                print(keys);
            else:
                pass
            if (groupName):
                group: Group = file.get(groupName)  # group2 = hf.get('group2/subfolder')
                items: List[Tuple] = list(
                    group.items())  # [(u'data3', <HDF5 dataset "data3": shape (100, 3333), type "<f8">)]
                if (items):
                    print(items)
                    try:
                        for i in range(len(items)):
                            print('recovering group class:', group.get(items[i][0]))
                            yield asarray(group.get(items[i][0]))  # n1 = group1.get('data1') \n np.array(n1).shape
                    except StopIteration as s:
                        cls.__closingH5PY(file)
                else:
                    pass
            elif (datasetNames):
                # cls.__closingH5PY(file)
                yield file.get(datasetNames)  # n1 =
    @staticmethod
    def __closingH5PY(file: File) -> None:
        file.close()




if __name__ == "__main__":
    pass