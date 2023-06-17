#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
from libs.PyImageAugmentation cimport *
from cython cimport boundscheck, wraparound, cdivision
from libcpp.vector cimport vector
from libc.string cimport memset, memcpy
from libc.stdlib cimport malloc, free
from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr, make_unique
from numpy cimport ndarray, uint8_t, import_array, float32_t
from numpy import  ascontiguousarray, uint8, dstack, ndarray, asarray, float32, zeros
from libc.stdlib cimport rand, RAND_MAX
ctypedef unsigned char uchar
import_array()

cdef class PyImageDataGenerator:
    cdef:
         uchar[:,:,:,:,:] final_images
         vector[Mat] augmentedImages
         int reserve_1, reserve_2, reserve_3
    def __cinit__(self, uchar[:,:,:,:,:]&image, double angle, int crop_w, int crop_h, float bright_alpha, int contrast, int noise_mean, float stdDev):
        self.reserve_1 = image.shape[0]
        self.reserve_2 = image.shape[1]
        self.reserve_3 = image.shape[2]
        self.final_images = image[:]
        self.display(self.final_images, angle, crop_w, crop_h, bright_alpha, contrast, noise_mean, stdDev)

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef void display(self, uchar[:,:,:,:,:]&image, double angle, int crop_w, int crop_h, float bright_alpha, int contrast, int noise_mean, float stdDev):
        cdef:
            int i, j, k
            Mat img
            vector[int] limits
            AugmentationManager * augmented
        for i in range(self.reserve_1):
            random_number = self.random_number(9)
            for j in range(self.reserve_2):
                for k in range(self.reserve_3):
                    img = self.np2Mat2D(image[i,j,k])
                    assert(img.rows ==image.shape[3] and img.cols==image.shape[4]), "the image dimensions are not identical to the dimensions of the original array"
                    augmented = new AugmentationManager(img, random_number, angle, crop_w, crop_h, bright_alpha, contrast, noise_mean, stdDev)
                    self.augmentedImages.push_back(augmented.getAugmentedImage(image.shape[3], image.shape[4]))
                    del augmented
        if(self.augmentedImages.size()>0):
            self.PyAugmentedImage(self.augmentedImages, image)
            self.augmentedImages.clear()

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef inline Mat np2Mat2D(self, uchar[:,:] image ):
        cdef ndarray[uint8_t, ndim=2, mode ='c'] np_buff = ascontiguousarray(image, dtype=uint8)
        cdef unsigned int* im_buff = <unsigned int*> np_buff.data
        cdef int r = image.shape[0]
        cdef int c = image.shape[1]
        cdef Mat m
        m.create(r, c, CV_8UC1)
        memcpy(m.data, im_buff, r*c)
        return m

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef inline int random_number(self, int ceiling)nogil:
        return <int>(rand()%ceiling) +1;

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef inline void Mat2np(self, Mat&m, uchar[:,:]&img_array):
        # Create buffer to transfer data from m.data
        cdef Py_buffer buf_info

        # Define the size / len of data
        cdef size_t len = m.rows*m.cols*m.elemSize()  #m.channels()*sizeof(CV_8UC3)

        # Fill buffer
        PyBuffer_FillInfo(&buf_info, NULL, m.data, len, 1, PyBUF_FULL_RO)

        # Get Pyobject from buffer data
        Pydata  = PyMemoryView_FromBuffer(&buf_info)

        # Create ndarray with data
        #print("channels ->", m.channels(), m.depth(), CV_32F)
        assert (m.channels()<2), "this function does not support images with 3 channels"
        if m.depth() == CV_32F :
            ary = ndarray(shape=(m.rows, m.cols), buffer=Pydata, order='c', dtype=float32)
        else :
            #8-bit image
            ary = ndarray(shape=(m.rows, m.cols), buffer=Pydata, order='c', dtype=uint8)

        cdef int i, j
        for i in range(m.rows):
            for j in range(m.cols):
                img_array[i,j] = ary[i,j]

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef inline void PyAugmentedImage(self, vector[Mat]images, uchar[:,:,:,:,:]&original):
        cdef:
            int i, j, k, m, n
            unsigned int  total, counter=0
            Mat img
            uchar[:,:] img_array = zeros((original.shape[3],original.shape[4]), dtype=uint8)
        total = self.reserve_1*self.reserve_2*self.reserve_3
        assert(images.size()== total), "the size of the vectors is not igual to dimensions of original array"
        for i in range(self.reserve_1):
            for j in range(self.reserve_2):
                for k in range(self.reserve_3):
                    img = images[counter]
                    if(img.rows==original.shape[3] and img.cols==original.shape[4]):
                        self.Mat2np(img, img_array)
                        #print("image ->", asarray(img_array).shape, counter)
                        if(counter<total):
                           counter+=1
                        else:
                            print("[Info]: Counter out of boundaries -> {} != {}".format(counter, total))
                            raise AssertionError()
                        for m in range(img.rows):
                            for n in range(img.cols):
                                original[i,j,k,m,n] = img_array[m,n]
                    else:
                        print("[Info]: false number of rows {} and columns {}".format(img.rows, img.cols))
                        raise AssertionError()
    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef inline vector[int] setLimits(self, int end, int steps):
        cdef:
            int i =0
            vector[int] arange
        for i in range(0,end, steps):
            arange.push_back(<int>i)
            # print("values ->", i)
        arange.push_back(end)
        return arange

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef inline bint compareInts(self, int index, vector[int]&limits) nogil:
        cdef int value
        if(limits.size()>0):
            value = limits[0]
            #print("values compared-> ", value , index)
            if(index==value):
                return True
            else:
                return False
        else:
            raise AssertionError()

    def getAugmentedImages(self):
        return asarray(self.final_images, dtype=float32)
