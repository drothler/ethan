#cython: language_level=3, wraparound=False, boundscheck=False, initializedcheck=False, nonecheck=False
import numpy as np
cimport numpy as np
import configparser
import cython
import sys
from helper cimport contains, get_binary_event
from helper import get_words_from_string

BOOL = np.uint8
ctypedef np.uint8_t BOOL_t

float32 = np.float32
ctypedef np.float32_t float32_t

cdef class FastTimeTable:

    # relevant objective node information
    cdef np.uint32_t id
    cdef np.uint8_t age
    cdef np.uint8_t days_since_infection
    cdef np.uint8_t status
    cdef np.uint8_t positive_test
    cdef np.uint8_t shows_symptoms
    cdef np.uint8_t is_quarantined

    # subjective properties
    cdef float32_t happiness, sociability, introvertedness

    # schedule information
    cdef np.ndarray schedule
    cdef np.uint64_t[:, :, :] schedule_view
    cdef np.uint8_t is_working, is_studying, has_home
    cdef np.uint16_t home_district, edu_district, work_district

    # array/memory view in bool type to tell whether person takes part in custom generated activities
    cdef np.ndarray takes_part_in_activities
    cdef np.ndarray other_facilities
    cdef np.ndarray environment

    cdef np.uint64_t home_facility, edu_facility, work_facility
    cdef np.ndarray transit_candidates
    cdef np.uint64_t[:, :] transit_candidate_view

    cdef int[:] personal_stats
    cdef str[:] attributes


    # memory view containing day and hour of time slots of flexible free time
    # necessary for table creation and updating after each
    cdef int[:, :] flexible_hours


    cdef fill_slot(self, Py_ssize_t day, Py_ssize_t hour, np.uint64_t slot_type, np.uint64_t generation_type, np.uint64_t facility_type, np.uint64_t facility_id)
    cdef void empty_slot(self, Py_ssize_t day, Py_ssize_t hour)
    cdef np.uint64_t[:] get_flexible_indices_on_day(self, Py_ssize_t day)
    cdef void prepare_personal_facilities(self, np.uint64_t[:] attribute_view)
    #cdef void set_transit_candidates(self, np.uint64_t[:, :] candidates)

    cdef np.uint64_t get_work_id(self)
    cdef np.uint64_t get_edu_id(self)
    cdef np.uint64_t get_home_id(self)

    cdef set_home_district(self, np.uint16_t district_id)
    cdef set_edu_district(self, np.uint16_t district_id)
    cdef set_work_district(self, np.uint16_t district_id)

    cdef np.uint8_t is_quarantined(self)

    cdef np.uint8_t is_working(self)
    cdef np.uint8_t is_studying(self)
    cdef np.uint8_t has_home(self)
    cdef void print_schedule(self)
    cdef void print_schedule_on_day(self, Py_ssize_t day)


cdef class FastFacilityTable:

    cdef np.ndarray schedule
    cdef str facility_name
    cdef np.uint64_t capacity, unique_id
    cdef np.uint32_t district_id
    cdef np.uint32_t[:, :, :] schedule_view

    cdef int get_size(self)
    cdef int get_id(self)
    cdef np.uint8_t add_node_at_time(self, Py_ssize_t day, Py_ssize_t hour, int id)
    cdef np.uint8_t remove_node_from_time(self, Py_ssize_t day, Py_ssize_t hour, int id)
    cdef void close_holes(self, Py_ssize_t day, Py_ssize_t hour, Py_ssize_t start_index)
    cdef void clear_schedule(self)
    cdef void print_schedule(self)
    cdef np.uint32_t[:] return_nodes_at_time(self, Py_ssize_t day, Py_ssize_t hour)
    cdef np.ndarray return_schedule(self)
    cdef np.uint64_t get_capacity(self)
    cdef int get_available_capacity(self, int day, int hour)
    cdef Py_ssize_t get_district(self)

cdef class FacilityManager:
    cdef FacilityManager facility_manager

    cdef np.ndarray facilities
    cdef FastFacilityTable[:, :] fac_view
    cdef np.ndarray facilities_by_district
    cdef np.uint64_t[:, :, :, :] fac_dis_view

    cdef np.ndarray infrastructure
    cdef FastFacilityTable[:, :] infrastructure_view
    cdef np.ndarray infrastructure_by_district
    cdef np.uint64_t[:, :, :, :] infrastructure_by_dis_view

    cdef np.ndarray general_pool

    cdef np.ndarray personal_facilities
    cdef FastFacilityTable[:, :, :] pers_view

    cdef str[:] fac_names, inf_names

    cdef dict facility_cfg, infrastructure_cfg, personal_cfg
    cdef int node_count, district_count
    cdef float32_t million_ratio

    cdef initialize_facilities(self)
    cdef void initialize_facility(self, int row, str name, int capacity_avg, float32_t capacity_sigma, int instances, int district_count)
    cpdef int[:, :, :, :] return_facilities_by_district(self)




    cdef initialize_personal_facilities(self, np.uint32_t[:, :, :, :] personal_facilities)
    cdef initialize_pers_fac_by_district(self)

    cdef initialize_infrastructures(self)
    cdef initialize_infrastructure(self, Py_ssize_t row, str name, int capacity_avg, int instances)
    cdef initialize_infrastructure_by_district(self)

    cdef np.uint64_t get_facility_for_district(self, Py_ssize_t type, Py_ssize_t district)
    cdef np.uint64_t get_infrastructure_for_district(self, Py_ssize_t type, Py_ssize_t district)

    cdef np.uint8_t add_node_to_facility(self, np.uint32_t node_id, Py_ssize_t generator, Py_ssize_t type, np.uint64_t fac_id, Py_ssize_t day, Py_ssize_t hour, Py_ssize_t district)
    cdef np.uint8_t remove_node_from_facility(self, np.uint32_t node_id, Py_ssize_t generator, Py_ssize_t type, np.uint64_t fac_id, Py_ssize_t day, Py_ssize_t hour, Py_ssize_t district)

    cpdef print_schedules(self)
    cdef print_personals(self)




cdef class HumanManager:

    cdef FacilityManager facility_manager

    cdef Py_ssize_t node_len
    # w - district id, x - facility type, y - facility index, z - id, capacity
    cdef np.uint32_t[:, :, :, :] graph_facility_view

    # assign id to index - usually just identity
    cdef  np.uint32_t[:] node_ids
    # x - node index (NOT NECESSARILY ID), y - connection index, z - (id, type)
    # first connection in each node column is (connection_count, 0)
    cdef  np.uint32_t[:, :, :] connections
    # x - node index, y - (home, work, edu) <- specified in cfg file (adaptable)
    cdef np.ndarray personal_facilities
    cdef np.uint64_t[:, :] pers_fac_view
    cdef np.ndarray timetables
    cdef object[:] timetable_view

    # personal generated attributes
    cdef np.ndarray introvertedness
    cdef np.float64_t[:] intro_view
    cdef np.ndarray ages
    cdef np.float64_t[:] age_view
    cdef np.ndarray happiness
    cdef int[:] happiness_view
    cdef np.ndarray node_district_info
    cdef np.uint16_t[:, :] node_district_info_view

    cdef np.ndarray work_info
    cdef int[:, :] work_view
    cdef np.ndarray edu_info
    cdef int[:, :] edu_view
    cdef np.ndarray transport_probabilities

    # regulatory attributes
    cdef int district_count
    cdef int max_work_hours
    cdef int work_avg, sleep_avg
    cdef int work_sigma, sleep_sigma
    cdef np.ndarray work_starts
    cdef int[:] work_start_view
    cdef np.ndarray work_starts_density

    cdef initialize_work_info(self)
    cdef np.uint32_t find_district_to_node(self, Py_ssize_t node_index, Py_ssize_t facility_type, np.uint32_t[:, :] distr_view, np.uint64_t[:, :] homes)
    #cdef initialize_edu_info(self)
    cpdef initialize_node_info(self, np.ndarray age_distr_array)




