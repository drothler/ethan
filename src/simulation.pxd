#cython: language_level=3, wraparound=False, boundscheck=False, initializedcheck=False, nonecheck=False, cdivision=True


import numpy as np
cimport numpy as np


# Naive simulator takes the pre-processed graph from DataLoader and outputs a set of encounter nodes
# encounter nodes are calculated with fixed properties based on the connection type
# any person will have a small probability to meet another person
# thus it is potentially possible for a person to have N encounters, although very very unlikely


BOOL = np.uint8
ctypedef np.uint8_t BOOL_t

float32 = np.float32
ctypedef np.float32_t float32_t

cdef extern from "math.h" nogil:
    float exp(float x)
    float log(float x)





cdef struct s_Encounter:
    int encounter_id       # unique id for each encounter
    int node_id            # id of encountered node
    int facility_id
    float32_t intensity, duration
    np.uint8_t day, hour

ctypedef s_Encounter CEncounter

cdef struct s_Activity:
    int activity_id, facility_per_million
    np.uint8_t start, end, environment, can_meet_connections, type
    bint seperate_pool
    float32_t participation_probability, week_frequency_avg, week_frequency_sigma, duration_avg, duration_sigma, capacity_avg, capacity_sigma

ctypedef s_Activity Activity

cdef struct s_Infrastructure:
    int infrastructure_id, instances_per_million
    float32_t population_percentage
    bint needs_schedule
ctypedef s_Infrastructure Infrastructure


cdef class FastSimulation:
    cdef int node_count
    cdef int[:] node_ids
    cdef int[:, :] facility_info
    cdef int[:, :, :] connections
    cdef Py_ssize_t activity_len, transit_len
    cdef np.ndarray facilities

    #cdef dict facilities



