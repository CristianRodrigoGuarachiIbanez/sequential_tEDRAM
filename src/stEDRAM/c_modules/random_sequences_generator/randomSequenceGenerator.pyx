from cython cimport boundscheck, wraparound, view
from numpy cimport int64_t, ndarray
from numpy import empty, random, asarray
from cpython.array cimport array
from array import array
from libc.string cimport memset
from libc.stdlib cimport malloc,free

ctypedef unsigned char uchar

cdef struct FREQUENCE:
    uchar key
    int value
ctypedef FREQUENCE freq


cdef class RandomSequencesGenerator:
    cdef:
        unsigned int rows,cols
        int **_matrix

    def __cinit__(self, unsigned int nr, unsigned int nc):
        self.rows = nr
        self.cols = nc
        self._matrix = <int**>malloc(nr*sizeof(int*))
        self.__fillUpMatrix()
        self.fillMatrixSequences()
        if(self._matrix == NULL):
            raise MemoryError

    def __deallocate__(self):
        free(self._matrix)

    cdef void __fillUpMatrix(self) nogil:
        cdef int i
        for i in range(self.rows):
            self._matrix[i] = <int*>malloc(self.cols*sizeof(int))

    @boundscheck(False)
    @wraparound(False)
    cdef void fillMatrixSequences(self) nogil:
        cdef:
            unsigned int N = self.rows;
            unsigned int M = self.cols;
            int i,j;
            int *sequences= self.fillArraySequences(self._matrix[0],M,0)
        #print(sequences)
        #memset(sequences,0,100*sizeof(int))
        #print('memset:',sequences)
        for i in range(N):
            sequences = self.fillArraySequences(self._matrix[i],M,i)
            #print(sequences)
            for j in range(M):
                self._matrix[i][j] = sequences[j]

    @boundscheck(False)
    @wraparound(False)
    cdef int*fillArraySequences(self, int arr[], unsigned int size, unsigned int index) nogil:
        cdef:
            unsigned int start = size * index
            unsigned int stop = size * (index +1);
            unsigned int i;
            unsigned counter = 0;
        for i in range(start,stop):
            arr[counter] = i;
            counter+=1;
        return arr

    @boundscheck(False)
    @wraparound(False)
    cdef list[list[int]] convert_to_python(self, int **ptr):
        cdef int i,j
        lst=[]

        for i in range(self.rows):
            lstIn=[]
            for j in range(self.cols):
                lstIn.append(ptr[i][j])
                #print(j, ptr[i][j])
            lst.append(lstIn)
            #print(i, lst[i])
        return lst
        
    def matrix(self):
        output = asarray(self.convert_to_python(self._matrix))
        random.shuffle(output)
        return output
