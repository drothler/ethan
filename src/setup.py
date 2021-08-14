from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
from Cython.Compiler.Options import get_directive_defaults

directive_defaults = get_directive_defaults()



setup(  name='ethan', 
        include_dirs=[np.get_include(), '.'], 
        ext_modules=cythonize(
                                [Extension(
                                    "scheduling", ["scheduling.pyx"],
                                    extra_compile_args=['-O3', '-march=native'],
                                    extra_link_args=['-O3', '-march=native']),
                                Extension(
                                    "simulation", ["simulation.pyx"],
                                    extra_compile_args=['-O3', '-march=native'],
                                    extra_link_args=['-O3', '-march=native']),
                                Extension(
                                    "helper", ["helper.pyx"],
                                    extra_compile_args=['-O3', '-march=native'],
                                    extra_link_args=['-O3', '-march=native']),
                                Extension(
                                    "disease", ["disease.pyx"],
                                    extra_compile_args=['-O3', '-march=native'],
                                    extra_link_args=['-O3', '-march=native'])]
        )
)