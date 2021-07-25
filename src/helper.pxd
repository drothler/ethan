#cython: language_level=3, wraparound=False, boundscheck=False, initializedcheck=False, nonecheck=False
import numpy as np
cimport numpy as np

float32 = np.float32
ctypedef np.float32_t float32_t

cdef bint contains(str string, str match)

cdef bint get_binary_event(float32_t probability)