#distutils: language = c++
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.utility cimport pair
cimport numpy as np
import numpy as np

# For cv::Mat usage
cdef extern from "opencv2/core/core.hpp":
  cdef int  CV_WINDOW_AUTOSIZE
  cdef int CV_8UC3
  cdef int CV_8UC1
  cdef int CV_32FC1
  cdef int CV_8U
  cdef int CV_32F

cdef extern from "opencv2/core/core.hpp" namespace "cv":
  cdef cppclass Mat:
    Mat() except +
    void create(int, int, int)
    void* data
    int rows
    int cols
    int channels()
    int depth()
    size_t elemSize()

# For Buffer usage
cdef extern from "Python.h":
    ctypedef struct PyObject
    object PyMemoryView_FromBuffer(Py_buffer *view)
    int PyBuffer_FillInfo(Py_buffer *view, PyObject *obj, void *buf, Py_ssize_t len, int readonly, int infoflags)
    enum:
        PyBUF_FULL_RO

cdef extern from "<algorithm>" namespace "std":
    Iter find_if[Iter, Func](Iter first, Iter last, Func pred)
    Iter std_remove "std::remove" [Iter, T](Iter first, Iter last, const T& val)

cdef extern from "../ImageAugmentation.h" namespace "img":
    cdef cppclass AugmentationManager:
        AugmentationManager(Mat &scr, int random_number, double angle, int crop_w, int crop_h, float bright_alpha, int contrast, int noise_mean, float stdDev) except+
        inline Mat getAugmentedImage(int rows, int cols)

        void rotation(Mat &scr, double angle );
        void flipping(Mat &scr, char direction);
        void shearing(const Mat&input, float Bx, float By);
        void cropping(Mat &image, const int cropSizeW, const int cropSizeH);
        void AddGaussianNoise(const Mat &image, double Mean, double StdDev);
        void contrast_brightness(Mat &image, double alpha, int beta);
        void algorithmSelector(Mat &image, int random_number, double angle, int crop_w, int crop_h, float bright_alpha, int contrast, int noise_mean, float stdDev);
        void shear(Mat&image)
        #int randNumberGenerator(int ceiling)
        Mat IMG;


cdef object Mat2np(Mat mat)