from cython_modules.img_random_sequences.randomSequencesConstructor import RandomSequencesConstructor
# from image_cython.image_splitter import SPLITTER
from numpy import ndarray, zeros, array
from typing import Generator, List
from pickle import load
from h5py import File
import sys
import logging
logger = logging.getLogger(__name__)
FORMAT = "%(filename)s:%(lineno)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT,
                    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler(sys.stdout)]
                    )

class SequenceConstructor:
    __matrix: ndarray
    _sequences: List[ndarray]

    def __init__(self, rows: int, cols: int, n_classes: int) -> None:
        '''
        create a matrix with a given number of rows and columns
        :param rows: integer, number of trials calculated from the total number of samples
        :param cols: integer, number of images in one image sequence
        '''
        randomSeq: RandomSequencesConstructor = RandomSequencesConstructor(rows//n_classes, cols)  # empty((rows//10, cols), dtype=int);
        self.__matrix = randomSeq.matrix()
        self._sequences: List[ndarray] = list()

    def getMatrix(self) -> ndarray:
        return self.__matrix

    def samples(self, features: ndarray, start: int, end: int) -> ndarray:
        listOfImgArraysSequences: List[ndarray] = list()
        currSequence: Generator = self.generateRandomIndexSequences(start, end)
        if len(self._sequences) != 0:
            self._sequences.clear()

        for i in range(start, end):
            current = next(currSequence)  # array of 10 values
            self._sequences.append(current)
            # print("Current shape: {} {}".format(current, features.shape))
            listOfImgArraysSequences.append(array(features[current, ...], dtype='float32'))
        logging.debug('saved indices in samples: {}'.format(self._sequences))
        return array(listOfImgArraysSequences, dtype='float32')

    def labels(self, labels: ndarray, start:int, end:int):
        assert len(self._sequences) == abs(start-end), ' the sequence list ist empty'
        listOfSequences: List[ndarray] = list()
        for i in range(len(self._sequences)):
            # print(self._sequences[i])
            listOfSequences.append(labels[self._sequences[i], ...])
        logging.debug('recovered indices in labels: {}'.format(self._sequences))
        return array(listOfSequences, dtype='int32')

    def generateRandomIndexSequences(self, start: int, end: int) -> Generator:
        '''
        returns the image sequences individually
        :param start: integer
        :param end: interger
        :return: generator with individual image sequences (one dime array)
        '''
        for i in range(start, end):
            yield self.__matrix[i]

if __name__ == '__main__':

    s = SequenceConstructor(100, 10, 7)  # 6
    with open('training_dataset/sample_disparity_maps.txt', 'rb') as file:
        data: ndarray = load(file)
        print("label shape", data.shape)
        counter: int = 0
        # for i in range(data.shape[0]):
        #     if (data[i, 3] == 1):
        #         counter += 1
        # print("counter",counter)
        b = s.samples(data, start=0, end=10)

    # print('a shape:',a.shape)
    print('b shape:', len(s.getMatrix()))

    pass