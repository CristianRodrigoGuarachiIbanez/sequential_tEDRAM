from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


setup(
    ext_modules=cythonize(Extension("randomSequenceGenerator",
        sources=["randomSequenceGenerator.pyx"],
        include_dirs=[numpy.get_include()]
        )
    )
)