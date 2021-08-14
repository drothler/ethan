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
    cdef np.uint8_t total_infection_time
    cdef np.uint8_t incubation_time
    cdef np.uint8_t status
    cdef np.uint8_t positive_test
    cdef np.uint8_t shows_symptoms
    cdef np.uint8_t is_showing_symptoms
    cdef np.uint8_t is_quarantined
    cdef np.uint8_t quarantine_span

    # subjective properties
    cdef float32_t sociability, introvertedness, happiness

    # schedule information
    cdef np.ndarray schedule, facility_ids, districts
    cdef np.uint32_t[:, :] facility_id_view
    cdef np.uint8_t[:, :, :] schedule_view
    cdef np.uint16_t[:, :] district_id_view
    cdef np.uint8_t is_working, is_studying, has_home
    cdef np.uint16_t home_district, edu_district, work_district

    # array/memory view in bool type to tell whether person takes part in custom generated activities
    cdef np.ndarray takes_part_in_activities
    cdef np.ndarray other_facilities
    cdef np.ndarray environment

    cdef np.uint32_t home_facility, edu_facility, work_facility
    cdef np.ndarray transit_candidates
    cdef np.uint32_t[:, :] transit_candidate_view
    cdef np.uint8_t transit_type

    cdef int[:] personal_stats
    cdef str[:] attributes


    # memory view containing day and hour of time slots of flexible free time
    # necessary for table creation and updating after each
    cdef int[:, :] flexible_hours

    cdef void fill_empty_slots(self)
    cdef void set_standard_transit(self, np.uint8_t type)
    cdef np.uint8_t[:] find_time_for_activity(self, np.uint8_t activity, np.uint8_t day, np.uint8_t hour_count, np.uint8_t[:, :] hour_restrictions)
    cdef fill_slot(self, Py_ssize_t day, Py_ssize_t hour, np.uint8_t slot_type, np.uint8_t generation_type, np.uint8_t facility_type, np.uint32_t facility_id, np.uint16_t district_id)
    cdef void empty_slot(self, Py_ssize_t day, Py_ssize_t hour)
    cdef np.uint8_t[:] get_flexible_indices_on_day(self, Py_ssize_t day)
    cdef np.uint8_t[:, :] find_flexible_timespans(self, np.uint8_t activity, np.uint8_t day, np.uint8_t hour_count, np.uint8_t[:, :] hour_restrictions)
    cdef void prepare_personal_facilities(self, np.uint32_t[:] attribute_view)
    #cdef void set_transit_candidates(self, np.uint32_t[:, :] candidates)

    cdef np.uint32_t get_work_id(self)
    cdef np.uint32_t get_edu_id(self)
    cdef np.uint32_t get_home_id(self)

    cdef np.uint8_t get_standard_transit(self)

    cdef np.uint16_t get_district_to_hour(self, Py_ssize_t day, Py_ssize_t hour)

    cdef void print_districts(self)
    cdef print_strings_on_day(self, Py_ssize_t day)

    cdef set_home_district(self, np.uint16_t district_id)
    cdef np.uint16_t get_home_district(self)
    cdef set_edu_district(self, np.uint16_t district_id)
    cdef np.uint16_t get_work_district(self)
    cdef set_work_district(self, np.uint16_t district_id)
    cdef np.uint16_t get_edu_district(self)

    cdef np.uint8_t is_quarantined(self)
    cdef void put_in_quarantine(self, np.uint8_t quarantine_time)
    cdef np.uint8_t update_status(self)
    cdef np.uint8_t[:] get_status(self)
    cdef void infect(self, np.uint8_t length, np.uint8_t symptoms, np.uint8_t incubation_period)
    cdef np.uint32_t[:] stay_home(self, Py_ssize_t day, Py_ssize_t hour)


    cdef np.uint8_t is_working(self)
    cdef np.uint8_t is_studying(self)
    cdef np.uint8_t has_home(self)
    cdef void print_schedule(self)
    cdef void print_schedule_on_day(self, Py_ssize_t day)


cdef class FastFacilityTable:

    cdef np.ndarray schedule
    cdef str facility_name
    cdef np.uint32_t capacity, unique_id
    cdef np.uint16_t district_id
    cdef np.uint32_t[:, :, :] schedule_view

    cdef int get_size(self)
    cdef np.uint32_t get_id(self)
    cdef np.uint8_t add_node_at_time(self, np.uint8_t day, np.uint8_t hour, np.uint32_t facility_id)
    cdef np.uint8_t remove_node_from_time(self, Py_ssize_t day, Py_ssize_t hour, np.uint32_t id)
    cdef void close_holes(self, Py_ssize_t day, Py_ssize_t hour, Py_ssize_t start_index)
    cdef void clear_schedule(self)
    cdef void print_schedule(self)
    cdef np.uint32_t[:] return_nodes_at_time(self, Py_ssize_t day, Py_ssize_t hour)
    cdef np.ndarray return_schedule(self)
    cdef np.uint32_t get_capacity(self)
    cdef np.uint32_t get_available_capacity(self, Py_ssize_t day, Py_ssize_t hour)
    cdef Py_ssize_t get_district(self)
    cdef np.uint8_t is_available(self, Py_ssize_t day, Py_ssize_t hour)

cdef class FacilityManager:

    cdef np.ndarray facilities
    cdef FastFacilityTable[:, :] fac_view
    cdef np.ndarray facilities_by_district
    cdef np.uint32_t[:, :, :, :] fac_dis_view

    cdef np.ndarray infrastructure
    cdef FastFacilityTable[:, :] infrastructure_view
    cdef np.ndarray infrastructure_by_district
    cdef np.uint32_t[:, :, :, :] infrastructure_by_dis_view

    cdef np.ndarray general_pool
    cdef np.uint8_t[:, :, :, :] general_view

    cdef np.ndarray personal_facilities
    cdef FastFacilityTable[:, :, :] pers_view

    cdef str[:] fac_names, inf_names

    cdef dict facility_cfg, infrastructure_cfg, personal_cfg
    cdef np.uint32_t node_count, district_count
    cdef float32_t million_ratio

    cdef initialize_facilities(self)
    cdef initialize_facility(self, Py_ssize_t row,  str name, np.uint32_t capacity_avg, np.float32_t capacity_sigma, np.uint32_t instances, np.uint32_t district_count)
    cpdef np.uint32_t[:, :, :, :] return_facilities_by_district(self)




    cdef initialize_personal_facilities(self, np.uint32_t[:, :, :, :] personal_facilities)
    cdef initialize_pers_fac_by_district(self)

    cdef initialize_infrastructures(self)
    cdef initialize_infrastructure(self, Py_ssize_t row, str name, np.uint32_t capacity_avg, np.uint32_t instances)
    cdef initialize_infrastructure_by_district(self)

    cdef np.uint32_t get_facility_for_district(self, Py_ssize_t type, Py_ssize_t district)
    cdef np.uint32_t get_infrastructure_for_district(self, Py_ssize_t type, Py_ssize_t district)

    cdef np.uint8_t add_node_to_facility(self, np.uint32_t node_id, Py_ssize_t generator, Py_ssize_t type, np.uint32_t fac_id, Py_ssize_t day, Py_ssize_t hour, Py_ssize_t district)
    cdef np.uint8_t remove_node_from_facility(self, np.uint32_t node_id, Py_ssize_t generator, Py_ssize_t type, np.uint32_t fac_id, Py_ssize_t day, Py_ssize_t hour, Py_ssize_t district)

    cdef np.uint8_t check_availability(self, Py_ssize_t day, Py_ssize_t hour, Py_ssize_t generator, Py_ssize_t type, np.uint32_t fac_id, Py_ssize_t district)
    cdef np.uint8_t check_availability_timespan(self, Py_ssize_t day, Py_ssize_t start, Py_ssize_t end, np.uint8_t generator, Py_ssize_t type, np.uint32_t fac_id, Py_ssize_t district)
    
    cdef np.uint8_t get_activity_count(self)
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
    cdef np.uint32_t[:, :] pers_fac_view
    cdef np.ndarray timetables
    cdef object[:] timetable_view

    # personal generated attributes
    cdef np.ndarray introvertedness
    cdef np.float32_t[:] intro_view
    cdef np.ndarray ages
    cdef np.float32_t[:] age_view
    cdef np.ndarray happiness
    cdef np.float32_t[:] happiness_view
    cdef np.ndarray node_district_info
    cdef np.uint16_t[:, :] node_district_info_view

    cdef np.ndarray work_info
    cdef np.uint32_t[:, :] work_view
    cdef np.ndarray edu_info
    cdef np.uint32_t[:, :] edu_view
    cdef np.ndarray transport_probabilities
    cdef np.ndarray transport_has_schedule
    cdef np.uint8_t[:] transport_has_schedule_view 

    cdef np.ndarray facility_probabilities
    cdef np.float32_t[:] facility_probability_view
    cdef np.ndarray activity_probabilities
    cdef np.float32_t[:, :, :] activity_probability_view
    cdef np.ndarray facility_frequency
    cdef np.float32_t[:] facility_frequency_view
    cdef np.ndarray facility_frequency_sigma
    cdef np.float32_t[:] facility_frequency_sigma_view
    cdef np.ndarray facility_types
    cdef np.uint8_t[:] facility_type_view
    cdef np.ndarray facility_flexibility
    cdef np.uint8_t[:] facility_flexibility_view
    cdef np.ndarray facility_sundays
    cdef np.uint8_t[:] facility_sunday_view
    cdef np.ndarray facility_duration
    cdef np.float32_t[:] facility_duration_view
    cdef np.ndarray facility_duration_sigma
    cdef np.float32_t[:] facility_duration_sigma_view
    cdef np.ndarray facility_constraints
    cdef np.uint8_t[:, :] fac_constraint_view
    cdef np.uint8_t[:] connection_facility

    # regulatory attributes
    cdef int district_count
    cdef int max_work_hours
    cdef int work_avg, sleep_avg
    cdef int work_sigma, sleep_sigma
    cdef np.ndarray work_starts
    cdef np.uint8_t[:] work_start_view
    cdef np.ndarray work_starts_density
    cdef np.uint8_t max_work_days
    cdef np.uint8_t max_halftime_days 
    cdef np.uint8_t sigma_halftime_days
    cdef np.uint8_t avg_work_days
    cdef np.uint8_t avg_halftime_days 
    cdef np.float32_t part_time_percentage






    cdef np.uint8_t school_day_start, school_max_end, school_hours_avg, school_hours_sigma, halftime_hours_avg, halftime_hours_sigma, max_halftime_hours

    cdef initialize_work_info(self)
    cpdef initialize_flexible_slots(self)

    cdef np.uint32_t find_district_to_node(self, Py_ssize_t node_index, Py_ssize_t facility_type, np.uint32_t[:, :] distr_view, np.uint32_t[:, :] homes)
    #cdef initialize_edu_info(self)
    cpdef initialize_node_info(self, np.ndarray age_distr_array)

    cdef void put_node_in_quarantine(self, np.uint32_t node_id, np.uint8_t time_span)
    cdef void infect_node(self, np.uint32_t node_id, np.uint8_t length, np.uint8_t symptoms, np.uint8_t incubation_period)
    cdef np.uint8_t[:] get_status_update(self, np.uint32_t node_id)
    cdef np.uint8_t[:, :] get_status_for_all(self)
    cdef np.uint8_t[:] update_nodes(self)

    

    cdef np.uint8_t schedule_appointment(self, np.uint8_t day, np.uint32_t node_one, np.uint32_t node_two, np.uint8_t generator, np.uint8_t desired_length)

    cdef np.uint8_t put_node_in_transit(self, np.uint32_t node_id, np.uint8_t day, np.uint8_t hour, np.uint8_t district)