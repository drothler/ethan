#cython: language_level=3, wraparound=False, boundscheck=False, initializedcheck=False, nonecheck=False
import cython
import numpy as np
cimport numpy as np

cdef bint contains(str string, str match):
    cdef char c
    cdef Py_ssize_t str_len = len(string)
    cdef Py_ssize_t match_len = len(match)
    cdef Py_ssize_t str_index = 0
    cdef Py_ssize_t match_count = 0
    for str_index in range(str_len):
        if string[str_index] == match[match_count]:
            match_count += 1
        if match_count == match_len:
            return True
    return False

cdef bint get_binary_event(np.float32_t probability):
    return probability > np.random.random()


def get_words_from_string(str string):
    cdef list words = (string.replace(' ', '')).split(',')

    return words

def get_ints_from_string(str string):
    cdef list words = (string.replace(' ', '')).split(',')
    for index, word in enumerate(words):
        words[index] = int(word)
    return words

def get_floats_from_string(str string):
    cdef list words = (string.replace(' ', '')).split(',')
    for index, word in enumerate(words):
        words[index] = np.float32(word)
    return words

def get_bools_from_string(str string):
    cdef list words = (string.replace(' ', '')).split(',')
    for index, word in enumerate(words):
        words[index] = (word=='True')
    return words
