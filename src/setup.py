from distutils.core import setup
from Cython.Build import cythonize
import numpy as np



setup(name='ethan', include_dirs=[np.get_include(), '.'], ext_modules=cythonize("*.pyx"))