# distutils: language = c++
from libc.stdlib cimport malloc, free
from numpy cimport ndarray, int64_t
from numpy import asarray, int64
from cython cimport boundscheck, wraparound
ctypedef unsigned char uchar
cdef class ReduceDimensions:
    cdef int* nulls
    cdef int* ones
    cdef:
        int s
    def __cinit__(self, int size):
        self.s = size
        self.nulls = <int*>malloc(self.s*sizeof(int))
        self.ones = <int*>malloc(self.s*sizeof(int))
        self.fillUpNulls()
        self.fillUpOnes()
        if(self.nulls is NULL or self.ones is NULL):
            raise MemoryError
    def __deallocate__(self):
        free(self.ones)
        free(self.nulls)
    @wraparound(False)
    @boundscheck(False)
    cdef void fillUpNulls(self) nogil:
        cdef:
            int i
        for i in range(self.s):
            #print("index",i)
            self.nulls[i]=0
    cdef void fillUpOnes(self ) nogil:
        cdef:
            int i
        for i in range(self.s):
            if(i==1):
                self.ones[i]=1
            else:
                self.ones[i]=-1
    @wraparound(False)
    @boundscheck(False)
    @staticmethod
    cdef ReduceDimensions reduceDimensions(int *nulls, int*ones, int size):
        """Factory function to create WrapperClass objects from
        given data type"""
        # Call to __new__ bypasses __init__ constructor
        cdef ReduceDimensions wrapper = ReduceDimensions.__new__(ReduceDimensions,size)
        wrapper.nulls = nulls
        wrapper.ones = ones
        #wrapper.fillUpNulls()
        #wrapper.fillUpOnes()
        return wrapper
    @wraparound(False)
    @boundscheck(False)
    @staticmethod
    cdef ReduceDimensions resize(int size):
        """Factory function to create WrapperClass objects with
        newly allocated my_c_struct"""
        cdef:
            int i
            int * array = <int *>malloc(size*sizeof(int))
        if array is NULL:
            raise MemoryError

        return ReduceDimensions.reduceDimensions(array, array, size)
    @wraparound(False)
    @boundscheck(False)
    cdef ndarray[int64_t, ndim=1] convertNulls(self, int size, int index):
        if(size>0):
            ReduceDimensions.resize(size)
            size = size
            print("size", size)
        else:#
            size = self.s
        output = []
        #print("size nulls", self.s)
        for i in range(size):
            if(i ==1):
                output.append(index)
            else:
                output.append(self.nulls[i])
        #print(output)
        return asarray(output, dtype=int64)
    @wraparound(False)
    @boundscheck(False)
    cdef ndarray[int64_t, ndim=1] convertOnes(self,int size=0):
        if(size>0):
            ReduceDimensions.resize(size)
            size = size
            #print("size", size)
        else:#
            size = self.s
        cdef int i
        output = []
        for i in range(size):
            output.append(self.ones[i])
        return asarray(output, dtype=int64)

    def get_nulls(self, size:int = 0, index:int =0):
        return self.convertNulls( size, index)
    def get_ones(self, size:int =0):
        return self.convertOnes(size)