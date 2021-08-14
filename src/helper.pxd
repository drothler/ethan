#cython: language_level=3, wraparound=False, boundscheck=False, initializedcheck=False, nonecheck=False
import numpy as np
cimport numpy as np



cdef bint contains(str string, str match)

cdef bint get_binary_event(np.float32_t probability)