%%cython

cimport cython

import numpy as np
from numpy cimport int64_t, int32_t, float64_t, ndarray
from libc.math cimport exp
cdef extern from "limits.h":
    int RAND_MAX
    
cdef class randomSequenceGenerator:
    cdef:
        unsigned int rows, cols
        #vector[narray] *_seq
        int64_t[:,:] _matrix
    def __cinit__(self, unsigned int nr, unsigned int nc):
        self.rows=nr
        self.cols=nc
        self._matrix = np.zeros((nr, nc), dtype=int)  
        self.randomSequences()
    def matrix(self):
        return np.array(self._matrix)
    cdef void randomSequences(self):
        cdef int64_t[:,:] seq = self.sequences()
        np.random.shuffle(seq)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ndarray[int64_t, ndim=2] shuffleSequences(self):
        cdef int64_t[:,:] seq = self.sequences()
        #print('vor Shufflen', np.array(seq))
        #cpdef long[:,:] sequ = np.asarray(seq)
        np.random.shuffle(seq)
        return np.asarray(seq); 
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ndarray[int64_t, ndim=2] sequences(self):
        cdef int N = self._matrix.shape[0]
        cdef int M = self._matrix.shape[1]
        cdef int64_t[:] sequence = np.zeros(M, dtype=int)

        cdef int  n, m

        for n in range(0, N):
            sequence = self._indexSequence(sequence,n)
            #self._seq.push_back(np.asarray(sequence))
            for m in range(0, M):
                self._matrix[n,m] = sequence[m]
        return np.asarray(self._matrix)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ndarray[int64_t, ndim=1] _indexSequence(self, int64_t[:] size, int index):

        cdef int start = size.shape[0] * index
        cdef int end = size.shape[0] * (index+1)
        cdef int i
        cdef int counter =0
        for i in range(start, end):
            size[counter] = i
            counter +=1
        return np.array(size)
