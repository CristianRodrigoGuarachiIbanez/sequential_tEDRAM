from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy
import sys
import os
import glob

import sys
import logging
logger = logging.getLogger(__name__)
FORMAT = "%(filename)s:%(lineno)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT,
                    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler(sys.stdout)]
                    )
lib_folder = os.path.join("/usr", 'lib', "x86_64-linux-gnu")

# Find opencv libraries in lib_folder

cvlibs = list()
for file in glob.glob(os.path.join(lib_folder, 'libopencv_*')):
    cvlibs.append(file.split('.')[0])

logging.debug(cvlibs)
cvlibs = ['-L{}'.format(lib_folder)] + ['opencv_{}'.format(lib.split(os.path.sep)[-1].split('libopencv_')[-1]) for lib in cvlibs]


logging.debug("LIBS {} {}".format(cvlibs, os.path.join("usr","include", "opencv2")))
logging.debug("FOLDER: {}".format(lib_folder, sys.prefix))
setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(Extension("batch_manager",
                                    sources=["batch_manager.pyx"],
                                    language="c++",extra_compile_args=["-std=c++17"],
                                    include_dirs=[numpy.get_include(),
                                                  os.path.join("/usr", 'include', 'opencv'),
                                                 ],
                                    library_dirs=[lib_folder, ],
                                    libraries=cvlibs,
                                    )
                          )
)