#cython: language_level=3, wraparound=False, boundscheck=False, initializedcheck=False, nonecheck=False, cdivision=True
from scheduling cimport HumanManager, FastFacilityTable
from data import DataLoader
import numpy as np
cimport numpy as np



cdef class FastSimulation:
    cdef np.uint8_t training
    cdef HumanManager human_manager
    cdef dict facility_dict
    cdef dict infrastructure_dict
    cdef dict personal_dict
    cdef dict legal_dict
    cdef dict population_dict
    cdef dict connection_dict
    cdef dict simulation_dict
    cdef dict type_dict
    cdef np.ndarray node_np, info_np
    cdef np.uint8_t[:, :] info_matrix
    cdef np.uint32_t[:, :] connected_encounter_ids
    cdef np.uint32_t[:] connected_encounters_facilities
    cdef np.uint8_t[:, :] connected_encounters_meta
    cdef np.uint8_t[:] connected_encounters_intensity
    cdef np.uint16_t[:] connected_encounters_duration

    cdef np.uint16_t day


    cdef void create_connected(self)
    cdef void create_scheduled(self)
    cdef void create_random(self)
    cdef void create_connected_encounters_for_node(self, Py_ssize_t node, np.uint8_t[:] count, np.uint8_t[:] intensity_avg,  np.uint8_t[:] intensity_sigma, np.uint16_t[:] duration_avg, np.uint16_t[:] duration_sigma)
    cdef np.uint32_t[:] get_nodes_to_type(self,np.uint32_t node, np.uint8_t type)