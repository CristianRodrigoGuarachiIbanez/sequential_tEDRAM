from libcpp.string cimport string
from libcpp cimport bool
ctypedef unsigned char uchar


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
    void release()
    Mat clone()
    void* data
    int rows
    int cols
    int channels()
    int depth()
    size_t elemSize()


#For cv::VideoCapture
cdef extern from "opencv2/videoio.hpp" namespace "cv":
    cdef cppclass VideoCapture:
        VideoCapture () except +
        VideoCapture (int index) except +
        VideoCapture (int index, int apiPreference) except +
        bool read (Mat image)
        bool isOpened()
        void release ()

# For Buffer usage
cdef extern from "Python.h":
    ctypedef struct PyObject
    object PyMemoryView_FromBuffer(Py_buffer *view)
    int PyBuffer_FillInfo(Py_buffer *view, PyObject *obj, void *buf, Py_ssize_t len, int readonly, int infoflags)
    enum:
        PyBUF_FULL_RO

cdef extern from "opencv2/core/types.hpp" namespace "cv":

    cdef cppclass Scalar_[T]:
        Scalar_() except +
        Scalar_(T v0) except +
        Scalar_(T v0, T v1, T v2=0, T v3=0) except +
    ctypedef Scalar_[int] Scalar

    cdef cppclass Point_[T]:
        Point_()  except +
        Point_(T x, T y)  except +
    ctypedef Point_[int] Point

    cdef cppclass Rect_[T]:
        Rect_() except +
        Rect_(T _x, T _y, T _width, T _height) except +
        Point br
        Point tl
    ctypedef Rect_[int] Rect

    cdef cppclass Range:
        Range() except +
        Range(int start, int end) except +