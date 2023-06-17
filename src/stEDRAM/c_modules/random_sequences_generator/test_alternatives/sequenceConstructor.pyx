from libc.stdlib cimport malloc, free
from numpy cimport ndarray, int64_t
from numpy import array, asarray, random
cdef class sequenceConstructor:
    cdef:
        unsigned int row, cols
        int**_matrix
    def __cinit__(self,unsigned int nr, unsigned int nc):
        self.rows = nr
        self.cols = nc
        self._matrix = (int **)malloc(nr * sizeof(int*));
        self.fillUpMatrix();
        if(self._matrix == NULL):
            raise MemoryError()
    def __dealloc__(self):
        if(self._matrix != NULL):
            free(self._matrix)
    cdef void fillUpMatrix(self):
        cdef int i;
        for i in range(len(nr)):
            self._matrix[i] = <int*>malloc(nc*sizeof(int))
    # --------------------check
    cdef int64_t[:,:] randomizeArraySequences(self): #ndarray[int64_t,dim=2]
        self.fillMatrixSequences();
        cdef int64_t[:,:] output = asarray(self._matrix))
        random.shuffle(output)
        return output
    # --------------------
    cdef void fillMatrixSequences(self):
        cdef:
            unsigned int N = self.rows;
            unsigned int M = self.cols;
            int i,j;
            int sequencesTemp[10] # new int[10];
        for i in range(N):
            sequencesTemp = self.fillArraySequences(self._matrix[i],M,i);
            for j in range(M):
                self._matrix[i][j] = sequencesTemp[j];

    cdef int*fillArraySequences(self, int arr[], unsigned int size, unsigned int index):
        cdef:
            unsigned int start = size * index
            unsigned int stop = size * (index +1);
            unsigned int i;
            unsigned counter = 0;
        for i in range(start,stop):
            arr[counter] = i;
            counter+=1;
        return arr


    @property
    def matrix(self):
        unsigned int i, j
        if(self._matrix!=NULL):
            for i in range(self.rows):
                for j in range(self.cols):
                    print(self._matrix[i][j])

