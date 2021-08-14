#cython: language_level=3, wraparound=True, boundscheck=True, initializedcheck=True, nonecheck=True, unraisable_tracebacks = False, cdivision=True

import numpy as np
cimport numpy as np
from helper cimport contains, get_binary_event
from helper import get_words_from_string, get_ints_from_string, get_floats_from_string, get_bools_from_string
import sys
import time
import csv
import gc
import line_profiler
import cython
from cpython.ref cimport PyObject
from pympler import asizeof

cdef class FastFacilityTable:


    def __init__(self, np.uint32_t unique_id, str facility_name, np.uint32_t capacity, np.uint16_t district_id):
        self.schedule = np.full([48, 7, capacity + 1], fill_value=-1, dtype=np.uint32)
        self.schedule_view = self.schedule
        self.schedule[:, :, capacity] = 0
        self.facility_name = facility_name
        self.capacity = capacity

        self.district_id = district_id
        self.unique_id = unique_id

    def __str__(self):
        return self.facility_name + str(self.schedule)

    cdef int get_size(self):
        return sys.getsizeof(self) + self.schedule.nbytes

    cdef np.uint32_t get_id(self):
        return self.unique_id


    cdef np.uint8_t add_node_at_time(self, np.uint8_t day, np.uint8_t hour, np.uint32_t facility_id):
        cdef np.uint32_t fill_count = 0
        fill_count = self.schedule_view[hour, day, self.capacity]

        # print(fill_count)
        if fill_count < self.capacity:
            self.schedule_view[hour, day, fill_count] = facility_id
            self.schedule_view[hour, day, self.capacity] += 1
            return 1
        else:
            return 0

    cdef np.uint8_t is_available(self, Py_ssize_t day, Py_ssize_t hour):
        cdef Py_ssize_t fill_count = self.schedule_view[hour, day, self.capacity]
        if fill_count < self.capacity:
            #print('check worked')
            return True
        else:
            return False

    cdef np.uint8_t remove_node_from_time(self, Py_ssize_t day, Py_ssize_t hour, np.uint32_t id):
        cdef Py_ssize_t index = 0
        for index in range(self.capacity):
            if self.schedule_view[hour, day, index] == id:
                self.schedule_view[hour, day, index] = -1
                self.schedule_view[hour, day, self.capacity] -= 1
                self.close_holes(day, hour, index)
                return True
        return False

    cdef void close_holes(self, Py_ssize_t day, Py_ssize_t hour, Py_ssize_t start_index):
        cdef Py_ssize_t index = 0
        if start_index < self.capacity - 1:
            for index in range(start_index, self.schedule_view[hour, day, self.capacity]):
                self.schedule_view[hour, day, index] = self.schedule_view[hour, day, index + 1]
            self.schedule_view[hour, day, index + 1] = -1

    cdef void clear_schedule(self):
        self.schedule = np.full([48, 7, self.capacity + 1], fill_value=-1, dtype=np.uint32)
        self.schedule_view = self.schedule
        self.schedule_view[:, :, self.capacity] = 0

    cdef void print_schedule(self):
        print(self.schedule)

    

    cdef np.uint32_t[:] return_nodes_at_time(self, Py_ssize_t day, Py_ssize_t hour):
        return self.schedule_view[hour, day]



    cdef np.ndarray return_schedule(self):
        return self.schedule

    cdef np.uint32_t get_capacity(self):
        return self.capacity

    cdef np.uint32_t get_available_capacity(self, Py_ssize_t day, Py_ssize_t hour):
        return self.capacity - self.schedule_view[day, hour, self.capacity]

    cdef Py_ssize_t get_district(self):
        return self.district_id

cdef class FastTimeTable:

    # schedule
    # each time slot has four entries
    # timeslot type - generation type - activity id - facility id
    # for example:
    # (1 - 0 - 1 - 1243) means:
    # routine - graph facility - edu - 1243

    # time slot types: (empty, flexible, routine, sleep, transit)
    # empty - for schedule creation and temporary filling
    # flexible - flex time, gets re-generated randomly each week
    # routine - work, edu at this time, is done every day, doesnt change from week to week
    # sleep - fixed, does not change based on any simulation input
    # transit - always before and after facility changes, should be automatically generated


    def __init__(self, np.uint32_t node_id, np.uint8_t age, np.float32_t introvertedness, np.float32_t happiness):


        self.id = node_id
        self.age = age

        # 0 - healthy, 1 - infected, 2 - cured, 3 - immune
        self.status = 0

        self.days_since_infection = -1
        self.total_infection_time = -1
        self.incubation_time = -1
        self.positive_test = False
        self.shows_symptoms = False
        self.is_showing_symptoms = False
        self.is_quarantined = False
        self.quarantine_span = -1

        self.introvertedness = introvertedness

        self.sociability = introvertedness * age
        self.happiness = happiness


        self.facility_ids = np.full([48, 7], fill_value=-1, dtype=np.uint32)
        self.facility_id_view = self.facility_ids

        self.schedule = np.full([48, 7, 3], fill_value=-1, dtype=np.uint8)
        self.schedule_view = self.schedule

        self.districts = np.zeros(shape=(48, 7), dtype=np.uint16)
        self.district_id_view = self.districts

        

        # personal attributes



        self.home_facility = -1
        self.has_home = False
        self.home_district = -1

        self.work_facility = -1
        self.is_working = False
        self.work_district = -1

        self.edu_facility = -1
        self.is_studying = False
        self.edu_district = -1



    cdef void fill_empty_slots(self):
        cdef Py_ssize_t day, hour
        for day in range(7):
            for hour in range(48):
                if self.schedule_view[hour, day, 0] == 255:
                    self.schedule_view[hour, day, 0] = 3
                    self.schedule_view[hour, day, 1] = 1
                    self.schedule_view[hour, day, 2] = 0
                    self.facility_id_view[hour, day] = self.home_facility

    cdef np.uint16_t get_district_to_hour(self, Py_ssize_t day, Py_ssize_t hour):
        return self.district_id_view[hour, day]

    cdef np.uint16_t get_home_district(self):
        return self.home_district

    cdef np.uint16_t get_work_district(self):
        return self.work_district

    cdef np.uint16_t get_edu_district(self):
        return self.edu_district

    cdef void set_standard_transit(self, np.uint8_t type):
        self.transit_type = type

    cdef np.uint8_t get_standard_transit(self):
        return self.transit_type

    cdef print_strings_on_day(self, Py_ssize_t day):
        cdef np.ndarray[ndim=1, dtype=object] output = np.ndarray(shape=(48), dtype=object)
        cdef str[:] output_view = output
        cdef Py_ssize_t hour
        for hour in range(48):
            if self.schedule_view[hour, day, 0] == 0:
                output_view[hour] = 'sleep'
            elif self.schedule_view[hour, day, 0] == 1:
                output_view[hour] = 'routine'
            elif self.schedule_view[hour, day, 0] == 2:
                output_view[hour] = 'transit'
            elif self.schedule_view[hour, day, 0] == 3:
                output_view[hour] = 'flexible'
            elif self.schedule_view[hour, day, 0] == 4:
                output_view[hour] = 'scheduled'
            elif self.schedule_view[hour, day, 0] == 5:
                output_view[hour] = 'meeting'
        print(output)

    cdef np.uint8_t[:, :] find_flexible_timespans(self, np.uint8_t activity, np.uint8_t day, np.uint8_t hour_count, np.uint8_t[:, :] hour_restrictions):

        cdef np.uint8_t[:] free_indices = self.get_flexible_indices_on_day(day)
        #print(np.asarray(free_indices))
        cdef Py_ssize_t index, max_index = free_indices.shape[0], min_index = 0
        cdef np.uint8_t slot_count = 1
        cdef np.uint8_t currently_in_chunk = True
        if free_indices.shape[0] != 0:
            while min_index < free_indices.shape[0] - 1 and free_indices[min_index] < hour_restrictions[0, activity]:
                min_index += 1
            #print('First available index:', free_indices[min_index], min_index)
            for index in range(min_index + 1, max_index):
                #print('index:', free_indices[index], index, currently_in_chunk, 'slot count:', slot_count)
                if free_indices[index] < hour_restrictions[1, activity]:

                    if free_indices[index] - 1 != free_indices[index - 1] and currently_in_chunk:
                        currently_in_chunk = False
                        slot_count += 1
                    else:
                        currently_in_chunk = True
                else:
                    break
            

        else:
            return None
                
        #print('Slot count', slot_count)
        
        cdef np.uint8_t[:, :] output = None
        cdef np.uint8_t tmp_length = 0
        if slot_count > 0:
            
            output = np.zeros(shape=(slot_count, 2), dtype=np.uint8)
            #print(np.asarray(output))
            #print(free_indices.shape[0], min_index)
            output[0, 0] = free_indices[min_index]
            output[0, 1] = 1
            tmp_length = 1
            slot_count = 0
            currently_in_chunk = True
            #print('First available index:', free_indices[min_index], min_index)
            for index in range(min_index + 1, max_index):
                #print('index:', free_indices[index], index, currently_in_chunk, 'slot count:', slot_count)
                if free_indices[index] < hour_restrictions[1, activity]:

                    if free_indices[index] - 1 != free_indices[index - 1] and currently_in_chunk:
                        currently_in_chunk = False
                        slot_count += 1
                        output[slot_count, 0] = free_indices[index]
                        output[slot_count, 1] = 1
                        
                    else:
                        output[slot_count, 1] += 1
                        currently_in_chunk = True
                else:
                    break
            #print(slot_count)
        #print(np.asarray(output))
        #print(min_index, max_index, slot_count)
        #print(free_indices[min_index], free_indices[max_index - 1])
        if free_indices is None or free_indices.shape[0] == 0:
            return None
        #if output[0, 0] + output[0, 1] > 47:
            #return np.full(shape=(3, 3), dtype=np.uint8, fill_value=-1)
        return output
                


    cdef np.uint8_t[:] find_time_for_activity(self, np.uint8_t activity, np.uint8_t day, np.uint8_t hour_count, np.uint8_t[:, :] hour_restrictions):
        #print(np.asarray(hour_restrictions))
        cdef np.uint8_t[:] free_indices = self.get_flexible_indices_on_day(day)
        cdef np.int8_t first_index
        cdef np.int8_t last_index
        cdef np.uint8_t max_length = 0
        cdef np.uint8_t first_max_index, last_max_index
        cdef np.uint8_t cur_length = 0
        cdef Py_ssize_t index, tmp_index
        cdef np.ndarray[ndim=1, dtype=np.uint8_t] return_array = np.zeros(shape=3, dtype=np.uint8)
        cdef np.uint8_t[:] return_view = return_array
        
        #input()
        if free_indices.shape[0] != 0:
            #input('Indices not empty')
            #print(np.asarray(free_indices))
            # skipping queue to get to opening indices
            index = 0
            #print(hour_restrictions[0, activity])
            #input('Getting to valid indices' )
            while free_indices[index] < hour_restrictions[0, activity] and index < free_indices.shape[0]:
                #print(free_indices[index])
                index +=1
            #print(free_indices[index], hour_restrictions[1, activity])
            # if the first available index is after shop closure
            if free_indices[index] > hour_restrictions[1, activity]:
                return_view[:] = 5
                return return_view 
            
            # setting pointer to first free hour and last free hour
            first_index = free_indices[index]
            last_index = free_indices[index]
            cur_length = 1
            max_length = 0
            #print(first_index, last_index, cur_length, max_length)
            #input()
            tmp_index = index
            # looping through indices
            # if indices are consecutive, they get treating as a block of free time
            for index in range(tmp_index, free_indices.shape[0]):
                #print(index, first_index, last_index, cur_length, max_length)
                #input()
                if free_indices[index] > hour_restrictions[1, activity]:
                    if cur_length > max_length:
                        first_max_index = first_index
                        last_max_index = last_index
                        max_length = cur_length
                    break

                elif free_indices[index] - 1 == last_index:
                    last_index = free_indices[index]
                    cur_length += 1
                    if cur_length == hour_count + 2:
                        return_view[0] = 1
                        return_view[1] = first_index
                        return_view[2] = last_index
                        return return_view
                else:
                    if cur_length > max_length:
                        first_max_index = first_index
                        last_max_index = last_index
                        max_length = cur_length
                    first_index = free_indices[index]
                    last_index = free_indices[index]

            if max_length >= 3 and last_max_index - first_max_index >= 2:
                return_view[0] = 2
                return_view[1] = first_max_index
                return_view[2] = last_max_index
            else:
                return_view[:] = 4

            return return_view



    cdef fill_slot(self, Py_ssize_t day, Py_ssize_t hour, np.uint8_t slot_type, np.uint8_t generation_type, np.uint8_t facility_type, np.uint32_t facility_id, np.uint16_t district_id):
        #if self.schedule_view[hour, day, 0] == 0:
           # raise ValueError('Can\'t overwrite fixed time slot on day', day, 'and hour', hour)
        #elif self.schedule_view[hour, day, 0] == 1:
            #raise ValueError('Can\'t overwrite routine time slot on day', day, 'and hour', hour, '\nFirst delete routine!', facility_type, facility_id, generation_type, slot_type)
        #elif self.schedule_view[hour, day, 0] == 2:
            #raise ValueError('Can\'t overwrite transit time slot on day', day, 'and hour', hour, '\nFirst delete transit!', facility_type, facility_id, generation_type, slot_type, np.asarray(self.schedule_view[hour, day, :]))
        #else:
        self.schedule_view[hour, day, 0] = slot_type
        self.schedule_view[hour, day, 1] = generation_type
        self.schedule_view[hour, day, 2] = facility_type
        self.facility_id_view[hour, day] = facility_id
        self.district_id_view[hour, day] = district_id

    cdef set_home_district(self, np.uint16_t district_id):
        self.home_district = district_id
        self.district_id_view[:, :] = district_id
    cdef set_edu_district(self, np.uint16_t district_id):
        self.edu_district = district_id
    cdef set_work_district(self, np.uint16_t district_id):
        self.work_district = district_id

    cdef void empty_slot(self, Py_ssize_t day, Py_ssize_t hour):
        if self.schedule_view[hour, day, 0] == 0:
            raise ValueError('Can\'t overwrite fixed time slot on day', day, 'and hour', hour)
        self.schedule_view[hour, day, :] = -1
        self.facility_id_view[hour, day] = -1

    cdef np.uint8_t[:] get_flexible_indices_on_day(self, Py_ssize_t day):
        cdef np.uint8_t index
        cdef np.uint8_t length = 0
        for index in range(self.schedule_view.shape[0]):
            if self.schedule_view[index, day, 0] == 3:
                length += 1
        
        cdef np.ndarray[ndim=1, dtype=np.uint8_t] indices = np.ndarray(shape=length, dtype=np.uint8)
        
        length = 0
        
        for index in range(self.schedule_view.shape[0]):
            if self.schedule_view[index, day, 0] == 3:
                indices[length] = index
                length += 1
            
        return indices

    # reads personal facilities from attribute_view and assigns them accordingly to home, edu and work

    cdef void prepare_personal_facilities(self, np.uint32_t[:] attribute_view):
        cdef Py_ssize_t facility_index

        self.home_facility = attribute_view[0]
        self.edu_facility = attribute_view[1]
        self.work_facility = attribute_view[2]

        if self.home_facility != 1:
            self.has_home = True
        if self.edu_facility != 1:
            self.is_studying = True
        if self.work_facility != 1:
            self.is_working = True



    cdef np.uint8_t is_working(self):
        if self.work_facility != np.uint32(-1):
            return True
        else:
            return False

    cdef np.uint8_t is_studying(self):
        return self.is_studying

    cdef np.uint8_t has_home(self):
        return self.has_home

    cdef void print_schedule(self):
        print(self.schedule)

    cdef void print_schedule_on_day(self, Py_ssize_t day):
        print(np.asarray(self.facility_id_view[:, day]).reshape(1, 48))

    cdef void print_districts(self):
        print(self.districts)

    cdef np.uint32_t get_work_id(self):
        return self.work_facility

    cdef np.uint32_t get_edu_id(self):
        return self.edu_facility

    cdef np.uint32_t get_home_id(self):
        return self.home_facility

    cdef np.uint8_t is_quarantined(self):
        return self.is_quarantined

    cdef np.uint32_t[:] stay_home(self, Py_ssize_t day, Py_ssize_t hour):
        cdef np.ndarray[ndim=1, dtype=np.uint32_t] fac_info = np.ndarray(shape=4, dtype=np.uint32)
        if self.schedule_view[hour, day, 0] != 0:
            fac_info[0] = self.schedule_view[hour, day, 1]
            fac_info[1] = self.schedule_view[hour, day, 2]
            fac_info[2] = self.facility_id_view[hour, day]
            fac_info[3] = self.district_id_view[hour, day]
            self.schedule_view[hour, day, 0] = 1
            self.schedule_view[hour, day, 1] = 0
            self.schedule_view[hour, day, 2] = 0
            self.facility_id_view[hour, day] = self.home_facility
            self.district_id_view[hour, day] = self.home_district
            return fac_info
        return None
        


    cdef void put_in_quarantine(self, np.uint8_t quarantine_time):
        if not self.is_quarantined:
            self.is_quarantined = True
            self.quarantine_span = quarantine_time
            
        else:
            self.is_quarantined = False
        
    cdef np.uint8_t update_status(self):
        if self.status == 1:
            self.days_since_infection += 1
            if self.days_since_infection > self.total_infection_time:
                self.infected = False
                self.is_showing_symptoms = False
                self.status = 2
            elif self.days_since_infection > self.incubation_time and self.shows_symptoms and not self.is_showing_symptoms:
                self.is_showing_symptoms = True

        return self.status
    
    cdef np.uint8_t[:] get_status(self):
        cdef np.uint8_t[10] output = [self.id, self.age, self.status, self.days_since_infection, self.total_infection_time, self.incubation_time, self.shows_symptoms, self.is_showing_symptoms, self.is_quarantined, self.positive_test]
        return output

    cdef void infect(self, np.uint8_t length, np.uint8_t symptoms, np.uint8_t incubation_period):
        if self.status == 0:
            self.status = 1
            self.total_infection_time = length
            self.shows_symptoms = symptoms
            self.incubation_time = incubation_period
            self.days_since_infection = 0

    


cdef class FacilityManager:

    def __init__(self, dict facilities, dict infrastructure, dict personals, np.uint32_t node_count, np.uint16_t district_count):
        self.facilities = None
        self.fac_view = None

        self.facilities_by_district = None
        self.fac_dis_view = None

        self.personal_facilities = None
        self.pers_view = None


        self.infrastructure = None
        self.infrastructure_view = None
        self.infrastructure_by_district = None
        self.infrastructure_by_dis_view = None

        self.facility_cfg = facilities
        self.infrastructure_cfg = infrastructure
        self.personal_cfg = personals
        self.node_count = node_count
        self.million_ratio = node_count / 1e6
        
        self.district_count = district_count
        self.general_pool = np.zeros(shape=(7, 48, self.district_count, node_count), dtype=np.uint8)
        self.general_view = self.general_pool
        print(asizeof.asizeof(self.general_pool))
        self.initialize_facilities()
        self.return_facilities_by_district()
        self.initialize_infrastructures()
        self.initialize_infrastructure_by_district()

    cdef initialize_infrastructure_by_district(self):
        cdef np.ndarray[ndim=4, dtype=np.uint32_t] inf_by_dis

        cdef FastFacilityTable tmpObj
        cdef Py_ssize_t inf_type, index, max_distr
        cdef Py_ssize_t shape, len, tmp_district
        shape = self.infrastructure.shape[0]
        len = self.infrastructure.shape[1]
        cdef np.ndarray[ndim=2, dtype=np.uint32_t] district_counts = np.zeros(shape=(shape, self.district_count), dtype=np.uint32)
        cdef np.uint32_t[:, :] district_view = district_counts

        for inf_type in range(shape):
            for index in range(len):
                tmpObj = self.infrastructure[inf_type][index]
                if tmpObj is None:
                    break
                else:
                    tmp_district = tmpObj.get_district()
                    district_view[inf_type, tmp_district] += 1

        max_distr = np.max(district_counts)

        inf_by_dis = np.zeros(shape=(self.district_count, shape, max_distr + 1, 2), dtype=np.uint32)
        #cdef np.ndarray count_tracker = np.zeros(shape=(self.district_count, 3), dtype=np.uint32)
        cdef Py_ssize_t fill_amount
        cdef np.uint32_t[:, :, :, :] inf_view = inf_by_dis
        inf_view[:, :, 0, 0] = 1
        inf_view[:, :, 0, 1] = 1
        for inf_type in range(shape):
            for index in range(len):
                tmpObj = self.infrastructure[inf_type][index]
                if tmpObj is None:
                    break
                else:
                
                    tmp_district = tmpObj.get_district()
                    #count_tracker[tmp_district, inf_type] += 1
                    fill_amount = inf_view[tmp_district, inf_type, 0, 0]
                    inf_view[tmp_district, inf_type, fill_amount, 0] = inf_type
                    inf_view[tmp_district, inf_type, fill_amount, 1] = index
                    inf_view[tmp_district, inf_type, 0, 0] += 1

        self.infrastructure_by_district = inf_by_dis
        self.infrastructure_by_dis_view = self.infrastructure_by_district
        #print(count_tracker)
        #print(self.infrastructure_by_district[0, 0, :, :])
        # print(np.asarray(inf_view[:, 2, :, 1]))

    cdef initialize_personal_facilities(self, np.uint32_t[:, :, :, :] personal_facilities):
        print('Generating graph-defined facilities from network...')
        cdef float start = time.time()

        cdef np.ndarray pers_spaces = np.array(get_bools_from_string(self.personal_cfg['personal_space']), dtype=bool)
        cdef np.uint8_t[:] pers_space_view = pers_spaces

        cdef np.ndarray pers_capacities = np.array(get_ints_from_string(self.personal_cfg['capacity']), dtype=np.uint32)
        cdef np.uint32_t[:] pers_capacity_view = pers_capacities

        cdef np.ndarray capacities = np.ndarray([self.district_count, personal_facilities.shape[1], personal_facilities.shape[2]], dtype=np.uint32)
        cdef np.uint32_t[:, :, :] capacity_view = capacities

        self.personal_facilities = np.ndarray([self.district_count, personal_facilities.shape[1], personal_facilities.shape[2]], dtype=object)

        cdef Py_ssize_t district_index, type_index, facility_index
        cdef FastFacilityTable tmp
        for district_index in range(self.district_count):
            for type_index in range(personal_facilities.shape[1]):
                for facility_index in range(personal_facilities[district_index, type_index, 0, 0]):
                    capacity_view[district_index, type_index, facility_index] = personal_facilities[district_index, type_index, facility_index + 1, 1] * pers_capacity_view[type_index]
                    tmp = FastFacilityTable(facility_name='test', unique_id=personal_facilities[district_index, type_index, facility_index + 1, 0], capacity=capacity_view[district_index, type_index, facility_index], district_id=district_index)
                    self.personal_facilities[district_index][type_index][facility_index] = tmp
                    


        self.pers_view = self.personal_facilities
        cdef float end = time.time()
        print('Initializing graph-defined facilities took', end-start, 'seconds')

    cdef initialize_facilities(self):
        print('Generating user-defined facilities from config file...')
        cdef float start = time.time()
        cdef list facility_types = get_words_from_string(self.facility_cfg['autogenerated_facilities'])

        cdef np.ndarray capacity_avgs = np.array(get_ints_from_string(self.facility_cfg['capacity_avg']), dtype=np.uint32)

        cdef np.ndarray capacity_sigmas = np.array(get_floats_from_string(self.facility_cfg['capacity_sigma']), dtype=np.float32)

        cdef np.ndarray activity_types = np.array(get_ints_from_string(self.facility_cfg['activity_type']), dtype=np.uint8)

        cdef np.ndarray facilities_per_million = np.array(get_ints_from_string(self.facility_cfg['facility_per_million']), dtype=np.uint32)

        cdef Py_ssize_t list_index
        cdef Py_ssize_t list_len = len(facility_types)

        cdef np.ndarray instances = facilities_per_million * self.million_ratio
        cdef np.ndarray seperate_pools = np.array(get_bools_from_string(self.facility_cfg['seperate_pools']), dtype=bool)

        instances = np.multiply(instances, seperate_pools)
        # cdef np.ndarray corrected_instances =
        self.facilities = np.ndarray([list_len, int(np.max(facilities_per_million))], dtype=object)
        for list_index in range(list_len):
            self.initialize_facility(list_index, facility_types[list_index], capacity_avgs[list_index], capacity_sigmas[list_index], <np.uint32_t>instances[list_index], self.district_count)
        cdef float end = time.time()
        self.fac_view = self.facilities
        print('Initializing user-defined facilities took', end-start, 'seconds')


    cdef initialize_facility(self, Py_ssize_t row,  str name, np.uint32_t capacity_avg, np.float32_t capacity_sigma, np.uint32_t instances, np.uint32_t district_count):

        cdef np.ndarray capacity = np.maximum(np.random.normal(loc=capacity_avg, scale=capacity_sigma, size=instances), 0).astype(np.float32)
        cdef np.ndarray district = np.random.randint(low=0, high=district_count, size=instances).astype(np.uint32)
        cdef np.float32_t[:] capacities = capacity
        cdef np.uint32_t[:] districts = district
        cdef np.uint32_t facility_id = 0
        cdef FastFacilityTable cur_facility
        for facility_id in range(instances):

            cur_facility = FastFacilityTable(unique_id=facility_id, facility_name=name, capacity=<np.uint32_t>capacities[facility_id], district_id=districts[facility_id])
            self.facilities[row][facility_id] = cur_facility


    cpdef np.uint32_t[:, :, :, :] return_facilities_by_district(self):
        cdef Py_ssize_t index, facility_id, max_district, activity_count, cur_index
        cdef np.uint32_t tmp_facility_id
        cdef np.ndarray facility_districts
        activity_count = self.facilities.shape[0]
        cdef np.ndarray district_counter = np.zeros([self.district_count, activity_count], dtype=np.uint32)
        cdef np.uint32_t[:, :] district_view = district_counter
        cdef Py_ssize_t tmp_district_id, tmp_activity_id, fill_amount
        cdef FastFacilityTable tmpTable

        for index in range(activity_count):
            for facility_id in range(self.facilities.shape[1]):
                if self.facilities[index][facility_id] is not None:
                    tmpTable = self.facilities[index][facility_id]
                    tmp_district_id = tmpTable.get_district()

                    district_view[tmp_district_id, index] += 1
        # print(district_counter)
        max_district = np.max(district_counter)
        facility_districts = np.full(shape=(self.district_count, activity_count, max_district + 1, 2), fill_value=-1, dtype=np.uint32)
        cdef np.uint32_t[:, :, :, :] view = facility_districts
        view[:, :, 0, 0] = 1
        view[:, :, 0, 1] = 1
        index = 0
        facility_id = 0
        for index in range(activity_count):
            for facility_id in range(self.facilities.shape[1]):
                if self.facilities[index, facility_id] is not None:
                    tmpTable = self.facilities[index][facility_id]
                    tmp_district_id = tmpTable.get_district()
                    tmp_facility_id = tmpTable.get_id()
                    fill_amount = view[tmp_district_id, index, 0, 0]
                    view[tmp_district_id, index, fill_amount, 0] = index
                    view[tmp_district_id, index, fill_amount, 1] = facility_id
                    view[tmp_district_id, index, 0, 0] += 1
                else:
                    break
        # print(facility_districts)
        self.facilities_by_district = facility_districts
        self.fac_dis_view = self.facilities_by_district
        return view

    cdef initialize_infrastructures(self):
        cdef list infrastructure_types = get_words_from_string(self.infrastructure_cfg['infrastructure_facilities'])
        cdef np.ndarray instances = np.array(get_floats_from_string(self.infrastructure_cfg['instances_per_million']), dtype=np.float32) * self.million_ratio
        cdef np.ndarray needs_schedule = np.array(get_bools_from_string(self.infrastructure_cfg['needs_schedule']), dtype=np.uint8)
        cdef np.ndarray capacities = np.array(get_ints_from_string(self.infrastructure_cfg['capacities_per_instance']), dtype=np.uint16)

        cdef Py_ssize_t index
        cdef Py_ssize_t inf_len = len(infrastructure_types)

        cdef np.uint32_t max_instance = <np.uint32_t>np.max(np.multiply(needs_schedule, instances))
        # print(np.multiply(needs_schedule, instances))


        self.infrastructure = np.ndarray([inf_len, max_instance], dtype=object)

        for index in range(inf_len):
            if needs_schedule[index]:
                self.initialize_infrastructure(index, infrastructure_types[index], capacities[index], instances[index])
        self.infrastructure_view = self.infrastructure

    cdef initialize_infrastructure(self, Py_ssize_t row, str name, np.uint32_t capacity_avg, np.uint32_t instances):
        cdef Py_ssize_t index

        cdef np.ndarray districts = np.random.randint(low=0, high=self.district_count, size=instances).astype(np.uint8)
        cdef np.uint8_t[:] district_view = districts
        cdef FastFacilityTable tmpObj
        # this will change with a future update
        # each public transport will get a set of districts it is available to
        # simulation will create a virtual railway system to allow for more consistency among public transport encounters
        for index in range(instances):
            tmpObj = FastFacilityTable(facility_name=name, unique_id=index, capacity=capacity_avg, district_id = district_view[index])
            self.infrastructure[row][index] = tmpObj
        self.infrastructure_view = self.infrastructure

    cpdef print_schedules(self):
        print(self.infrastructure.shape[0], self.infrastructure.shape[1])
        cdef FastFacilityTable tmp = self.infrastructure[0, 10]
        print(self.infrastructure)

    cdef print_personals(self):
        cdef FastFacilityTable tmp = self.personal_facilities[0, 0, 0]
        tmp.print_schedule()
        print(tmp.get_id(), tmp.get_capacity())

    cdef initialize_pers_fac_by_district(self):
        pass

    cdef np.uint32_t get_facility_for_district(self, Py_ssize_t fac_type, Py_ssize_t district):
        cdef np.uint32_t random_index, index, fac_id
        cdef FastFacilityTable tmpObj

        

        if self.fac_dis_view[district, fac_type, 0, 0] == 1:
            #print(np.asarray(self.fac_dis_view[district, fac_type, :, :]))
            return np.uint32(-1)

        
        random_index = self.fac_dis_view[district, fac_type, 0, 1]
            
        self.fac_dis_view[district, fac_type, 0, 1] = (self.fac_dis_view[district, fac_type, 0, 1] % (self.fac_dis_view[district, fac_type, 0, 0] - 1)) + 1
        
        index = self.fac_dis_view[district, fac_type, random_index, 1]
            
        tmpObj = self.fac_view[fac_type, index]

        #if not tmpObj.is_available()

        
            
        fac_id = tmpObj.get_id()
        
        return fac_id

    cdef np.uint32_t get_infrastructure_for_district(self, Py_ssize_t inf_type, Py_ssize_t district):
        cdef np.uint32_t random_index, index, inf_id
        cdef FastFacilityTable tmpObj

        if self.infrastructure_by_dis_view[district, inf_type, 0, 0] == 1:
            return np.uint32(-1)

        
        random_index = self.infrastructure_by_dis_view[district, inf_type, 0, 1]
            
        self.infrastructure_by_dis_view[district, inf_type, 0, 1] = (self.infrastructure_by_dis_view[district, inf_type, 0, 1] % (self.infrastructure_by_dis_view[district, inf_type, 0, 0] - 1)) + 1
        
        index = self.infrastructure_by_dis_view[district, inf_type, random_index, 1]
            
        tmpObj = self.infrastructure_view[inf_type, index]

        #if not tmpObj.is_available()

        
            
        inf_id = tmpObj.get_id()
        
        return inf_id


    cdef np.uint8_t check_availability_timespan(self, Py_ssize_t day, Py_ssize_t start, Py_ssize_t end, np.uint8_t generator, Py_ssize_t type, np.uint32_t fac_id, Py_ssize_t district):
        cdef FastFacilityTable tmpObj
        cdef Py_ssize_t index, tmp_index, hour_index
        cdef np.uint32_t tmp_id

        if generator == 0:
            for index in range(self.pers_view[district, type].shape[0]):
                tmpObj = self.pers_view[district, type, index]
                tmp_id = tmpObj.get_id()
                if fac_id == tmp_id:
                    for hour_index in range(start, end + 1):
                        if not tmpObj.is_available(day=day, hour=hour_index):
                            print('Not available on day', day, 'and hour', hour_index)
                            return False
                    return True
                
        elif generator == 1:
            for index in range(1, self.fac_dis_view[district, type, 0, 0]):
                tmp_index = self.fac_dis_view[district, type, index, 1]
                tmpObj = self.fac_view[type, tmp_index]
                tmp_id = tmpObj.get_id()
                if fac_id == tmp_id:
                    for hour_index in range(start, end + 1):
                        if not tmpObj.is_available(day=day, hour=hour_index):
                            print('Not available on day', day, 'and hour', hour_index)
                            return False
                    return True
        else:
            for index in range(1, self.infrastructure_by_dis_view[district, type, 0, 0]):
                tmp_index = self.infrastructure_by_dis_view[district, type, index, 1]
                tmpObj = self.infrastructure_view[type, tmp_index]
                tmp_id = tmpObj.get_id()

                if fac_id == tmp_id:
                    for hour_index in range(start, end + 1):
                        if not tmpObj.is_available(day=day, hour=hour_index):
                            print('Not available on day', day, 'and hour', hour_index)
                            return False
                    return True
        return False


    cdef np.uint8_t check_availability(self, Py_ssize_t day, Py_ssize_t hour, Py_ssize_t generator, Py_ssize_t type, np.uint32_t fac_id, Py_ssize_t district):
        
        cdef FastFacilityTable tmpObj
        cdef Py_ssize_t index, tmp_index 
        cdef np.uint32_t tmp_id
        if generator == 0:
            for index in range(self.pers_view[district, type].shape[0]):
                tmpObj = self.pers_view[district, type, index]
                tmp_id = tmpObj.get_id()
                if fac_id == tmp_id:
                    return tmpObj.is_available(day=day, hour=hour)
        elif generator == 1:
            for index in range(1, self.fac_dis_view[district, type, 0, 0]):
                tmp_index = self.fac_dis_view[district, type, index, 1]
                tmpObj = self.fac_view[type, tmp_index]
                tmp_id = tmpObj.get_id()
                if fac_id == tmp_id:
                    return tmpObj.is_available(day=day, hour=hour)
        else:
            for index in range(1, self.infrastructure_by_dis_view[district, type, 0, 0]):
                tmp_index = self.infrastructure_by_dis_view[district, type, index, 1]
                #print(tmp_index)
                tmpObj = self.infrastructure_view[type, tmp_index]
                #print('Managed to create Object')
                #print(tmpObj)
                tmp_id = tmpObj.get_id()
                if fac_id == tmp_id:
                    #print(tmpObj.get_available_capacity(day=day, hour=hour))
                    return tmpObj.is_available(day=day, hour=hour)

        return False
    
    cdef np.uint8_t get_activity_count(self):
        return self.facilities.shape[0]

    cdef np.uint8_t add_node_to_facility(self, np.uint32_t node_id, Py_ssize_t generator, Py_ssize_t fac_type, np.uint32_t fac_id, Py_ssize_t day, Py_ssize_t hour, Py_ssize_t district):
        cdef Py_ssize_t index
        # 0 means personal facilities (school, work, edu)

        # 1 means config facilities

        # 2 means infrastructure

        cdef np.uint32_t tmp_type, tmp_index, tmp_id
        cdef FastFacilityTable tmpObj = None

        #with open('../resources/district_overview.csv', 'w') as csv_out:
            #csv_writer = csv.writer(csv_out)
            #csv_writer.writerow(['facility', 'district'])
            #for index in range(self.district_count):
                #for row in range(self.personal_facilities[index, 0].shape[0]):
                    #tmpObj = self.personal_facilities[index, 0, row]
                    #if tmpObj is not None:
                        #temp_id = tmpObj.get_id()
              
                        #csv_writer.writerow([temp_id, index])
        
        if generator == 0:

            for index in range(self.pers_view[district, fac_type].shape[0]):
                tmpObj = self.pers_view[district, fac_type, index]
                tmp_id = tmpObj.get_id()
                if tmp_id == fac_id:
                    #return_value = tmpObj.add_node_at_time(day, hour, node_id)
                    #self.pers_view[district, fac_type, index] = tmpObj
                    return tmpObj.add_node_at_time(day, hour, node_id)
            #print('object not found, should never happen')
            return False


        # generated facilities
        elif generator == 1:
            for index in range(1, self.fac_dis_view[district, fac_type, 0, 0] + 1):
                tmp_type = self.fac_dis_view[district, fac_type, index, 0]
                tmp_index = self.fac_dis_view[district, fac_type, index, 1]
                tmpObj = self.fac_view[tmp_type, tmp_index]
                if tmpObj is not None:
                    tmp_id = tmpObj.get_id()
                    if tmp_id == fac_id:
                        return tmpObj.add_node_at_time(day, hour, node_id)
                        
                else:
                    break
            #print('object not found, should never happen')
            return False
        elif generator == 2:
            for index in range(1, self.infrastructure_by_dis_view[district, fac_type, 0, 0]):
                tmp_type = self.infrastructure_by_dis_view[district, fac_type, index, 0]
                tmp_index = self.infrastructure_by_dis_view[district, fac_type, index, 1]
                tmpObj = self.infrastructure_view[tmp_type, tmp_index]
                if tmpObj is not None:
                    #print('Obj is not empty')
                    tmp_id = tmpObj.get_id()
                    if tmp_id == fac_id:
                        #print(tmp_id, 'is found')
                        return tmpObj.add_node_at_time(day, hour, node_id)
                        # self.infrastructure_view[tmp_type, tmp_index] = tmpObj
                        
                else:
                    break
        
            #print('object not found, should never happen', index, self.infrastructure_by_dis_view[district, fac_type, 0, 0])
            #print(district, day, hour, fac_id, np.asarray(self.infrastructure_by_dis_view[district, fac_type, :, 1]))
            return False
        else:
            self.general_view[day, hour, district, node_id] = True
            return True


    cdef np.uint8_t remove_node_from_facility(self, np.uint32_t node_id, Py_ssize_t generator, Py_ssize_t type, np.uint32_t fac_id, Py_ssize_t day, Py_ssize_t hour, Py_ssize_t district):
        cdef Py_ssize_t index
        # 0 means personal facilities (school, work, edu)

        # 1 means config facilities

        # 2 means infrastructure

        cdef np.uint32_t tmp_type, tmp_index, tmp_id
        cdef FastFacilityTable tmpObj

        cdef np.uint8_t return_value = False
        if generator == 0:

            for index in range(self.pers_view[district, type].shape[0]):

                tmpObj = self.pers_view[district, type, index]
                if tmpObj is not None:
                    tmp_id = tmpObj.get_id()
                    if tmp_id == fac_id:
                        return_value = tmpObj.remove_node_from_time(node_id, day, hour)
                        return return_value

                else:
                    break
            #print('This shouldnt happen lol')
            return False

        # generated facilities
        elif generator == 1:
            for index in range(self.fac_dis_view[district, type, 0, 0]):
                tmp_type = self.fac_dis_view[district, type, index + 1, 0]
                tmp_index = self.fac_dis_view[district, type, index + 1, 1]
                tmpObj = self.fac_view[tmp_type, tmp_index]
                if tmpObj is not None:
                    tmp_id = tmpObj.get_id()
                    if tmp_id == fac_id:
                        return_value = tmpObj.remove_node_from_time(node_id, day, hour)
                        return return_value
                else:
                    break
           # print('This should also not happen lol')
            return False
        elif generator == 2:
            for index in range(self.infrastructure_by_dis_view[district, type, 0, 0]):
                tmp_type = self.infrastructure_by_dis_view[district, type, index + 1, 0]
                tmp_index = self.infrastructure_by_dis_view[district, type, index + 1, 1]
                tmpObj = self.infrastructure_view[tmp_type, tmp_index]
                if tmpObj is not None:
                    tmp_id = tmpObj.get_id()
                    if tmp_id == fac_id:
                        return_value = tmpObj.remove_node_from_time(node_id, day, hour)
                        return return_value
                else:
                    break
            #print('why does this happen? You have only yourself to blame')
            return False
        else:
            self.general_view[day, hour, district, node_id] = False
            return True



cdef class HumanManager:

    def __init__(self, np.uint32_t[:] node_ids, np.uint32_t[:, :, :] connections, np.ndarray personal_facilities, dict legal_dict, dict population_dict, dict facilities, dict infrastructure, dict personals, dict simulation_dict, np.uint32_t districts):
        self.node_ids = node_ids
        self.connections = connections
        self.personal_facilities = personal_facilities
        self.pers_fac_view = self.personal_facilities
        self.node_len = node_ids.shape[0]
        self.district_count = districts
        # initializing an instance of facility manager automatically creates all required config facilities
        self.facility_manager = FacilityManager(facilities, infrastructure, personals, self.node_len, districts)
        # Auto Facilities - Graph Facilities - Node Time Tables - Fill Node Tables
        #     YES                 NO                 NO                NO

        self.timetables = np.ndarray([self.node_len], dtype=object)
        self.timetable_view = self.timetables

        # filler array, i want to feed the data loader age pyramid data in the future
        self.ages = np.minimum(np.maximum(np.random.normal(size=self.node_len, loc=45, scale=10), 100), 0).astype(np.float32)
        self.age_view = self.ages
        self.introvertedness = np.minimum(np.maximum(np.random.normal(size=self.node_len, loc=0.5, scale=0.25), 1), 0).astype(np.float32)
        self.intro_view = self.introvertedness
        self.happiness = None
        self.happiness_view = None

        self.work_info = np.ndarray([self.node_len, 3], dtype=np.uint32)
        self.work_view = self.work_info
        self.edu_info = np.ndarray([self.node_len, 3], dtype=np.uint32)
        self.edu_view = self.edu_info

        
        self.node_district_info = np.full(shape=(self.node_len, 3), fill_value=-1, dtype=np.uint16)
        self.node_district_info_view = self.node_district_info

        self.max_work_hours = int(legal_dict['max_work_hours'])
        self.work_avg = int(population_dict['average_fulltime_work'])
        self.work_sigma = int(population_dict['sigma_fulltime_work'])
        self.work_starts_density = np.array(get_floats_from_string(population_dict['work_start_times_density']))

        self.max_work_days = np.uint8(legal_dict['max_work_days'])
        self.max_halftime_days = np.uint(legal_dict['max_halftime_days'])
        self.avg_work_days = self.max_work_days - 1
        self.avg_halftime_days = np.uint8(population_dict['average_halftime_work'])
        self.sigma_halftime_days = np.uint8(population_dict['sigma_halftime_work'])
        self.part_time_percentage = np.float32(population_dict['part_time_percentage'])


        self.max_halftime_hours = np.uint8(legal_dict['max_halftime_hours'])
        self.halftime_hours_avg = np.uint8(population_dict['average_halftime_hours'])
        self.halftime_hours_sigma = np.uint8(population_dict['sigma_halftime_hours'])
        self.work_starts = np.array(get_ints_from_string(population_dict['work_start_times'])).astype(np.uint8)
        self.work_start_view = self.work_starts

        self.school_day_start = np.uint8(legal_dict['school_start'])
        self.school_max_end = np.uint8(legal_dict['school_max_end'])
        self.school_hours_avg = np.uint8(legal_dict['school_hours_avg'])
        self.school_hours_sigma = np.uint8(legal_dict['school_hours_sigma'])

        self.sleep_avg = int(simulation_dict['sleep_hours_default'])
        self.sleep_sigma = int(simulation_dict['sleep_hours_sigma'])

        self.transport_probabilities = np.array(get_floats_from_string(infrastructure['population_percentage']))
        self.transport_probabilities = np.append(self.transport_probabilities, 1 - np.sum(self.transport_probabilities))
        self.transport_probabilities = self.transport_probabilities / np.sum(self.transport_probabilities)

        self.facility_probabilities = np.array(get_floats_from_string(facilities['participation_probability']), dtype=np.float32)
        self.facility_probability_view = self.facility_probabilities
        self.facility_frequency = np.array(get_floats_from_string(facilities['frequency_avg'])).astype(np.float32)
        self.facility_frequency_view = self.facility_frequency
        self.facility_frequency_sigma = np.array(get_floats_from_string(facilities['frequency_sigma'])).astype(np.float32)
        self.facility_frequency_sigma_view = self.facility_frequency_sigma
        self.facility_types = np.array(get_ints_from_string(facilities['activity_type'])).astype(np.uint8)
        self.facility_type_view = self.facility_types
        self.facility_flexibility = np.array(get_bools_from_string(facilities['flexible']))
        self.facility_flexibility_view = self.facility_flexibility
        self.facility_sundays = np.array(get_bools_from_string(facilities['includes_sunday']))
        self.connection_facility = np.array(get_bools_from_string(facilities['can_meet_connections'])).astype(np.uint8)
        self.facility_constraints = 2 * np.array((np.array(get_ints_from_string(facilities['open_hour'])), np.array(get_ints_from_string(facilities['close_hour'])))).astype(np.uint8)
        self.fac_constraint_view = self.facility_constraints
        self.facility_sunday_view = self.facility_sundays
        self.activity_probabilities = None
        self.activity_probability_view = None

        self.facility_duration = np.array(get_floats_from_string(facilities['duration_avg'])).astype(np.float32)
        self.facility_duration_view = self.facility_duration
        self.facility_duration_sigma = np.array(get_floats_from_string(facilities['duration_sigma'])).astype(np.float32)
        self.facility_duration_sigma_view = self.facility_duration_sigma

        self.transport_has_schedule = np.append(np.array(get_bools_from_string(infrastructure['needs_schedule'])), False).astype(np.uint8)
        self.transport_has_schedule_view = self.transport_has_schedule
        # Calling this creates all necessary
        self.initialize_work_info()
        #profile = line_profiler.LineProfiler(self.initialize_node_info)
        #profile.runcall(self.initialize_node_info, np.full(fill_value=0.01, dtype=np.float32, shape=100))
        #profile.print_stats()
        #self.initialize_node_info(self.ages)
        # Auto Facilities - Graph Facilities - Node Time Tables - Fill Node Tables
        #     YES                 YES                 YES                NO



    cdef initialize_work_info(self):
        gc.collect()
        print('Initializing Node Information (districts, work/home/edu places etc)')
        cdef Py_ssize_t node_index, type_index, facility_index
        cdef Py_ssize_t work_len = 0
        cdef Py_ssize_t work_avg, work_sigma
        cdef np.ndarray[ndim=2, dtype=np.float32_t] work_length = np.minimum(np.maximum(np.random.normal(loc=self.work_avg, scale=self.work_sigma, size=(self.node_len, 7)), 1), self.max_work_hours).astype(np.float32)
        cdef np.float32_t[:, :] work_length_view = work_length


        cdef np.ndarray work_ids, work_counts, edu_ids, edu_counts, home_ids, home_counts
        work_ids, work_counts = np.unique(self.personal_facilities[:, 2], return_counts=True)
        home_ids, home_counts = np.unique(self.personal_facilities[:, 0], return_counts=True)
        edu_ids, edu_counts = np.unique(self.personal_facilities[:, 1], return_counts=True)

        cdef np.ndarray[ndim=2, dtype=np.uint32_t] work_places, edu_places, home_places
        cdef np.ndarray[ndim=3, dtype=np.uint32_t] places
        work_places = np.stack((work_ids, work_counts)).astype(np.uint32)
        edu_places = np.stack((edu_ids, edu_counts)).astype(np.uint32)
        home_places = np.stack((home_ids, home_counts)).astype(np.uint32)

        cdef Py_ssize_t max_dim = np.max(np.array([work_places.shape[1], home_places.shape[1], edu_places.shape[1]], dtype=np.uint32))

        places = np.ndarray(shape=(3, max_dim, 2), dtype=np.uint32)
        cdef np.uint32_t[:, :, :] place_view = places

        for node_index in range(home_places.shape[1]):
            place_view[0, node_index, 0] = home_places[0, node_index]
            place_view[0, node_index, 1] = home_places[1, node_index]

        for node_index in range(edu_places.shape[1]):
            place_view[1, node_index, 0] = edu_places[0, node_index]
            place_view[1, node_index, 1] = edu_places[1, node_index]

        for node_index in range(work_places.shape[1]):
            place_view[2, node_index, 0] = work_places[0, node_index]
            place_view[2, node_index, 1] = work_places[1, node_index]

        print(home_places)

        cdef np.ndarray[ndim=2, dtype=np.uint32_t] fac_districts = (np.random.randint(low=0, high=self.district_count, size=(3, max_dim))).astype(np.uint32)
        cdef np.ndarray _, district_counts
        cdef Py_ssize_t maximum_district_dim = 0, tmp
        for node_index in range(3):

            _, district_counts = np.unique(fac_districts[node_index], return_counts=True)
            tmp = np.max(district_counts)
            if tmp > maximum_district_dim:
                maximum_district_dim = tmp


        fac_districts = np.hstack((np.zeros(shape=(3, 1), dtype=np.uint32), fac_districts))

        cdef np.uint32_t[:, :] distr_view = fac_districts
        fac_districts[0, 0] = home_places.shape[1] - 1
        fac_districts[1, 0] = edu_places.shape[1] - 1
        fac_districts[2, 0] = work_places.shape[1] - 1



        #
        cdef np.ndarray id_districts = np.ndarray(shape=(self.node_len), dtype=np.uint32)
        cdef np.uint32_t[:] node_districts = id_districts

        cdef np.uint32_t facility_id
        for node_index in range(self.node_len):
            self.node_district_info_view[node_index, 0] = self.find_district_to_node(node_index, 0, distr_view, home_places)
            if self.pers_fac_view[node_index, 1] != 18446744073709551615:
                self.node_district_info_view[node_index, 1] = self.find_district_to_node(node_index, 1, distr_view, edu_places)
            if self.pers_fac_view[node_index, 2] != 18446744073709551615:
                self.node_district_info_view[node_index, 2] = self.find_district_to_node(node_index, 2, distr_view, work_places)



        # order by district, facility type, facility index, (facility_id, facility_capacity)
        cdef np.ndarray[ndim=4, dtype=np.uint32_t] pers_fac_array = np.ndarray(shape=(self.district_count, 3, maximum_district_dim + 1, 2), dtype=np.uint32)

        cdef np.uint32_t[:, :, :, :] personal_facilities_view = pers_fac_array
        personal_facilities_view[:, :, 0, 0] = 0
        cdef Py_ssize_t max_index
        cdef np.uint32_t capacity, unique_id, district
        for type_index in range(3):
            max_index = fac_districts[type_index, 0]
            for facility_index in range(1, max_index + 2):

                unique_id = place_view[type_index, facility_index - 1, 0]

                capacity = place_view[type_index, facility_index - 1, 1]
                district = fac_districts[type_index, facility_index]
                personal_facilities_view[district, type_index, personal_facilities_view[district, type_index, 0, 0] + 1, 0] = unique_id
                personal_facilities_view[district, type_index, personal_facilities_view[district, type_index, 0, 0] + 1, 1] = capacity
                personal_facilities_view[district, type_index, 0, 0] += 1
                # print('Adding facility of type', type_index, 'with id:', unique_id, 'to district:', district, 'capacity:', capacity)

        # print(pers_fac_array)
        self.facility_manager.initialize_personal_facilities(personal_facilities_view)

        

        # self.facility_manager.print_personals()


    cpdef initialize_flexible_slots(self):
        #gc.collect()
        print('Filling Time Tables with flexible user-defined activities...')
        start = time.time()

        cdef Py_ssize_t index, activity, day, activity_hour, slot_index, starting_index, activity_start_index

        cdef np.uint8_t activity_count = self.facility_manager.get_activity_count()


        cdef np.ndarray[ndim=2, dtype=np.uint8_t] activity_participation = np.random.binomial(n=1, p=self.facility_probabilities, size=(self.node_len, activity_count)).astype(np.uint8)
        cdef np.uint8_t[:, :] activity_participation_view = activity_participation
        cdef np.ndarray[ndim=2, dtype=np.uint8_t] activity_frequency = np.minimum(6, np.maximum(0, np.random.normal(loc=self.facility_frequency, scale=self.facility_frequency_sigma, size=(self.node_len, activity_count)))).astype(np.uint8)
        
        cdef np.ndarray[ndim=3, dtype=np.float32_t] activity_probabilities = np.full(shape=(self.node_len, activity_count, 7), fill_value=0.14285714, dtype=np.float32)
        print(activity_probabilities[0])
        activity_probabilities[:, :, 6] = (activity_probabilities[:, :, 6] * self.facility_sundays)
        activity_probabilities[:, :, :-1] = activity_probabilities[:, :, :-1] + np.array([(1 - self.facility_sundays) * 0.02380952,]*6).T
        activity_probabilities = activity_probabilities * np.array([activity_frequency.T, ] * 7).T
        print(activity_probabilities[0])
        self.activity_probabilities = activity_probabilities
        self.activity_probability_view = self.activity_probabilities        
        cdef np.ndarray[ndim=3, dtype=np.uint8_t] activity_days = np.random.binomial(n=1, p=activity_probabilities, size=(self.node_len, activity_count, 7)).astype(np.uint8)
        cdef np.uint8_t[:, :, :] activity_day_view = activity_days
        print(np.array([self.facility_duration, ]*7).T, np.array([self.facility_duration_sigma, ]*7).T)
        cdef np.ndarray[ndim=3, dtype=np.uint8_t] activity_hours = np.minimum(48, np.maximum(1, np.random.normal(loc=np.array([self.facility_duration, ]*7).T, scale=np.array([self.facility_duration_sigma, ]*7).T, size=(self.node_len, activity_count, 7)))).astype(np.uint8)
        cdef np.uint8_t[:, :, :] activity_hour_view = activity_hours
        print(activity_hours)
        #print(activity_participation)
        #print(activity_frequency)
        activity_days = activity_days * np.array([activity_participation.T, ]*7).T
        cdef np.uint32_t max_value = np.uint32(-1)
        cdef FastTimeTable table
        cdef np.uint8_t[:] activity_info
        cdef np.uint8_t[:, :] potential_slots
        cdef np.uint32_t to_transit, from_transit
        cdef np.uint32_t to_transit_id = max_value, from_transit_id = max_value
        cdef np.uint8_t transit_type
        cdef np.uint16_t tmp_district_to, tmp_district_from

        cdef np.uint32_t activity_facility
        cdef np.uint16_t counter = 0
        cdef np.uint64_t success_counter = 0
        cdef np.uint64_t fail_counter = 0
        cdef np.uint64_t success_counter_fac = 0
        cdef np.uint64_t fail_counter_fac = 0

        cdef np.uint8_t[:] random_offsets = np.random.randint(low=0, high=128, dtype=np.uint8, size=128)
        cdef Py_ssize_t offset_index = 0
        cdef np.uint8_t[:] slot_offsets = np.random.randint(low=0, high=128, dtype=np.uint8, size=128)
        cdef Py_ssize_t slot_offset = 0
        #print(activity_days[0])
        #print(self.timetables)
        counter = 0
        for index in range(self.node_len):
            #print(index)
            table = self.timetable_view[index]
            transit_type = table.get_standard_transit()
            to_transit_id = max_value
            from_transit_id = max_value
            #table.print_districts()
            
            for activity in range(activity_count):
                if activity_participation_view[index, activity]:
                    for day in range(7):
                        #table.print_strings_on_day(day)
                        to_transit_id = max_value
                        from_transit_id = max_value
                        if activity_day_view[index, activity, day] != 0:
                            potential_slots = table.find_flexible_timespans(activity, day, activity_hour_view[index, activity, day], self.fac_constraint_view)
                            
                            #print(potential_slots)
                            if potential_slots is not None:
                                slot_index = random_offsets[offset_index] % potential_slots.shape[0]
                                activity_info = potential_slots[slot_index]
                                
                                if activity_info[1] > activity_hour_view[index, activity, day] + 2:
                                    activity_start_index = activity_info[0] + (slot_offsets[slot_offset] % (activity_info[1] - activity_hour_view[index, activity, day] - 2))
                                    to_transit = activity_start_index
                                    from_transit = activity_start_index + activity_hour_view[index, activity, day] + 1
                                    slot_offset = (slot_offset + 1) % 128
                                    offset_index = (offset_index + 1) % 128
                                    #print('Start_index', to_transit, 'End_index', from_transit, np.asarray(activity_info), np.asarray(potential_slots))
                                    #table.print_strings_on_day(day)
                                    tmp_district_to = table.get_district_to_hour(day, (to_transit - 1 + 48)%48)
                                    tmp_district_from = tmp_district_to

                                    activity_facility = self.facility_manager.get_facility_for_district(type=activity, district=tmp_district_from)
                                    counter = 0
                                    while activity_facility == max_value and counter<self.district_count:
                                        activity_facility = self.facility_manager.get_facility_for_district(type=activity, district=(tmp_district_from + 1) % self.district_count)
                                        tmp_district_from = (tmp_district_from + 1) % self.district_count
                                        counter += 1
                                    if activity_facility == max_value:
                                        #print(activity, tmp_district_from)
                                        #print(activity_facility)
                                        print('failure')
                                    else:
                                        
                                        if self.facility_manager.check_availability_timespan(day=day, start=from_transit+1, end=to_transit-1, generator=1, type=activity, fac_id=activity_facility, district=tmp_district_from):
                                            for activity_hour in range(to_transit+1, from_transit):
                                                self.facility_manager.add_node_to_facility(index, 1, activity, activity_facility, day, activity_hour, tmp_district_from)
                                                
                                                table.fill_slot(day, activity_hour, 4, 1, activity, activity_facility, tmp_district_from)
                                                
                                            if not self.transport_has_schedule_view[transit_type]:
                                                table.fill_slot(day, to_transit, 2, 2, transit_type, self.node_ids[index], tmp_district_to)
                                                table.fill_slot(day, from_transit, 2, 2, transit_type, self.node_ids[index], tmp_district_from)
                                            else:
                                                counter = 0
                                                while to_transit_id == max_value and counter <=3:
                                                    to_transit_id = self.facility_manager.get_infrastructure_for_district(type=transit_type, district=tmp_district_to)
                                                    #print(tmp_district_to, 'District inf found', to_transit_id)
                                                    #print(from_transit)
                                                    if not self.facility_manager.check_availability(day, to_transit, 2, transit_type, to_transit_id, tmp_district_to):
                                                        #print('Transit not available, WHY?')
                                                        to_transit_id = max_value
                                                    counter += 1
                                                #print(from_transit)
                                                if to_transit_id == max_value:
                                                    table.fill_slot(day=day, hour=to_transit, slot_type=2, generation_type=3, facility_type=self.transport_probabilities.shape[0]-1, facility_id=self.node_ids[index], district_id = tmp_district_to)
                                                    self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=3, type=transit_type, fac_id=fac_id, day=day, hour=to_transit, district=tmp_district_to)
                                                else:
                                                    fac_id = to_transit_id
                                                    table.fill_slot(day=day, hour=to_transit, slot_type=2, generation_type=2, facility_type=transit_type, facility_id=fac_id, district_id=tmp_district_to)
                                                    self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=2, type=transit_type, fac_id=fac_id, day=day, hour=to_transit, district=tmp_district_to)
                                                while to_transit_id == max_value and counter <=3:
                                                    to_transit_id = self.facility_manager.get_infrastructure_for_district(type=transit_type, district=tmp_district_from)
                                                        #print(tmp_district_to, 'District inf found', to_transit_id)
                                                        #print(from_transit)
                                                    if not self.facility_manager.check_availability(day, from_transit, 2, transit_type, from_transit_id, tmp_district_from):
                                                            #print('Transit not available, WHY?')
                                                        from_transit_id = max_value
                                                    counter += 1
                                                    #print(from_transit)
                                                if from_transit_id == max_value:
                                                    table.fill_slot(day=day, hour=from_transit, slot_type=2, generation_type=3, facility_type=self.transport_probabilities.shape[0]-1, facility_id=self.node_ids[index], district_id = tmp_district_from)
                                                    self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=3, type=transit_type, fac_id=fac_id, day=day, hour=from_transit, district=tmp_district_from)
                                                else:
                                                    fac_id = from_transit_id
                                                    table.fill_slot(day=day, hour=from_transit, slot_type=2, generation_type=2, facility_type=transit_type, facility_id=fac_id, district_id=tmp_district_from)
                                                    self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=2, type=transit_type, fac_id=fac_id, day=day, hour=from_transit, district=tmp_district_from)
                                        else:
                                            print('failure')
                                            print('failure')

                                elif activity_info[1] >= 3:
                                    activity_start_index = activity_info[0]
                                    to_transit = activity_start_index
                                    from_transit = activity_start_index + activity_info[1] + 1
                                    slot_offset = (slot_offset + 1) % 128
                                    offset_index = (offset_index + 1) % 128
                                    #print(to_transit, from_transit, np.asarray(activity_info))
                                    tmp_district_to = table.get_district_to_hour(day, (to_transit - 1 + 48)%48)
                                    tmp_district_from = tmp_district_to

                                    activity_facility = self.facility_manager.get_facility_for_district(type=activity, district=tmp_district_from)
                                    counter = 0
                                    while activity_facility == max_value and counter<self.district_count:
                                        activity_facility = self.facility_manager.get_facility_for_district(type=activity, district=(tmp_district_from + 1) % self.district_count)
                                        tmp_district_from = (tmp_district_from + 1) % self.district_count
                                        counter += 1
                                    if activity_facility == max_value:
                                        print('failure to find activity facility')
                                        #print(activity, tmp_district_from)
                                        #print(activity_facility)
                                    else:
                                        if self.facility_manager.check_availability_timespan(day=day, start=from_transit+1, end=to_transit-1, generator=1, type=activity, fac_id=activity_facility, district=tmp_district_from):
                                            for activity_hour in range(to_transit+1, from_transit):
                                                self.facility_manager.add_node_to_facility(index, 1, activity, activity_facility, day, activity_hour, tmp_district_from)
                                                table.fill_slot(day, activity_hour, 4, 1, activity, activity_facility, tmp_district_from)
                                                if not self.transport_has_schedule_view[transit_type]:
                                                    table.fill_slot(day, to_transit, 2, 1, transit_type, self.node_ids[index], tmp_district_to)
                                                    table.fill_slot(day, from_transit, 2, 1, transit_type, self.node_ids[index], tmp_district_from)
                                                else:
                                                    counter = 0
                                                    while to_transit_id == max_value and counter <=3:
                                                        to_transit_id = self.facility_manager.get_infrastructure_for_district(type=transit_type, district=tmp_district_to)
                                                        #print(tmp_district_to, 'District inf found', to_transit_id)
                                                        #print(from_transit)
                                                        if not self.facility_manager.check_availability(day, to_transit, 2, transit_type, to_transit_id, tmp_district_to):
                                                            #print('Transit not available, WHY?')
                                                            to_transit_id = max_value
                                                        counter += 1
                                                    #print(from_transit)
                                                    if to_transit_id == max_value:
                                                        table.fill_slot(day=day, hour=to_transit, slot_type=2, generation_type=3, facility_type=self.transport_probabilities.shape[0]-1, facility_id=self.node_ids[index], district_id = tmp_district_to)
                                                        self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=3, type=self.transport_probabilities.shape[0]-1, fac_id=fac_id, day=day, hour=to_transit, district=tmp_district_to)
                                                    else:
                                                        fac_id = to_transit_id
                                                        table.fill_slot(day=day, hour=to_transit, slot_type=2, generation_type=1, facility_type=transit_type, facility_id=fac_id, district_id=tmp_district_to)
                                                        self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=2, type=transit_type, fac_id=fac_id, day=day, hour=to_transit, district=tmp_district_to)
                                                    while to_transit_id == max_value and counter <=3:
                                                        to_transit_id = self.facility_manager.get_infrastructure_for_district(type=transit_type, district=tmp_district_from)
                                                            #print(tmp_district_to, 'District inf found', to_transit_id)
                                                            #print(from_transit)
                                                        if not self.facility_manager.check_availability(day, from_transit, 2, transit_type, from_transit_id, tmp_district_from):
                                                                #print('Transit not available, WHY?')
                                                            from_transit_id = max_value
                                                        counter += 1
                                                        #print(from_transit)
                                                    if from_transit_id == max_value:
                                                        table.fill_slot(day=day, hour=from_transit, slot_type=2, generation_type=1, facility_type=self.transport_probabilities.shape[0]-1, facility_id=self.node_ids[index], district_id = tmp_district_from)
                                                        self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=3, type=transit_type, fac_id=fac_id, day=day, hour=from_transit, district=tmp_district_from)
                                                    else:
                                                        fac_id = from_transit_id
                                                        table.fill_slot(day=day, hour=from_transit, slot_type=2, generation_type=1, facility_type=transit_type, facility_id=fac_id, district_id=tmp_district_from)
                                                        self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=2, type=transit_type, fac_id=fac_id, day=day, hour=from_transit, district=tmp_district_from)
                                        else:
                                            print('failure')

                                    

                                else:
                                    #print(np.asarray(potential_slots))
                                    #print(np.asarray(activity_info))
                                    #table.print_strings_on_day(day)
                                    counter += 1
                                    slot_offset = (slot_offset + 1) % 128
                                    offset_index = (offset_index + 1) % 128
                                    #print('Didnt work, strange')
                                
                                #to_transit = activity_info[activity_start_index]
                                #from_transit = activity_info[activity_start_index + activity_hour_view[index, activity, day]]
                                #print(to_transit, from_transit)
                            
                            
                            
                            


        end = time.time()
        #print(success_counter, fail_counter, success_counter_fac, fail_counter_fac)
        print('Succesfully filled schedules with flexible activities in ', end-start, 'seconds')

    

    cpdef initialize_node_info(self, np.ndarray age_distr_array):

        gc.collect()

        # creating ages based on distribution
        cdef np.ndarray[ndim=1, dtype=np.uint8_t] age_array = np.random.choice(a=np.arange(age_distr_array.shape[0]), size=self.node_len, p=age_distr_array).astype(np.uint8)
        cdef np.uint8_t[:] age_view = age_array

        # creating social scores
        cdef np.ndarray[ndim=1, dtype=np.float32_t] social_scores = np.minimum(np.maximum(np.random.normal(loc=5, scale=2.5, size=self.node_len), 0), 10).astype(np.float32)
        cdef np.float32_t[:] social_view = social_scores

        # creating happiness factor among population
        cdef np.ndarray[ndim=1, dtype=np.float32_t] happiness = np.minimum(np.maximum(np.random.normal(loc=5, scale=2.5, size=self.node_len), 0), 10).astype(np.float32)
        self.happiness = happiness
        self.happiness_view = self.happiness


        


        cdef Py_ssize_t index
        cdef FastTimeTable table

        # how long does each person sleep
        cdef np.ndarray[ndim=1, dtype=np.int8_t] sleep_hours = (2 * np.minimum(np.maximum(np.random.normal(loc=self.sleep_avg, scale=self.sleep_sigma, size=self.node_len), 5), 10)).astype(np.int8)
        cdef np.int8_t[:] sleep_view = sleep_hours

        # how long would each person work full time (from 3 to max_work_hours, normal distribution)
        cdef np.ndarray[ndim=1, dtype=np.int8_t] work_hours = (2 * np.minimum(np.maximum(np.random.normal(loc=self.work_avg, scale=self.work_sigma, size=self.node_len), 3), self.max_work_hours)).astype(np.int8)
        cdef np.int8_t[:] work_view = work_hours
        cdef np.ndarray[ndim=1, dtype=np.uint8_t] halftime_work_hours = (2 * np.minimum(np.maximum(np.random.normal(loc=self.halftime_hours_avg, scale=self.halftime_hours_sigma, size=self.node_len), 1), self.max_halftime_hours)).astype(np.uint8)
        cdef np.uint8_t[:] halftime_view = halftime_work_hours



        # when does each person start working each day
        cdef np.ndarray[ndim=1, dtype=np.int8_t] work_starting_times = (2 * np.random.choice(a=self.work_starts, p=self.work_starts_density, size=self.node_len)).astype(np.int8)
        cdef np.int8_t[:] work_start_times_view = work_starting_times

        # how long it take each person to get ready for work (from 60 to 90 minutes)
        cdef np.ndarray[ndim=1, dtype=np.int8_t] ready_times = (np.random.randint(low=2, high=3, size=self.node_len)).astype(np.int8)
        cdef np.int8_t[:] ready_time_view = ready_times

        cdef np.ndarray[ndim=2, dtype=np.uint8_t] part_time = np.tile(np.random.binomial(n=1, p=self.part_time_percentage, size=self.node_len), (7, 1)).astype(np.uint8)
        print(np.asarray(part_time))

        #cdef np.uint8_t[:, :] work_day_count = np.maximum(4)
        #cdef np.uint8_t[:, :] fulltime_hours = np.minimum(self.max_work_hours, np.maximum(6, np.random.normal(loc=self.work_avg, scale=self.work_sigma, size=(7, self.node_len)))).astype(np.uint8)
        #cdef np.uint8_t[:, :] halftime_hours = np.minimum(self.max_halftime_hours, np.maximum(2, np.random.normal(loc=self.halftime_hours_avg, scale=self.halftime_hours_sigma, size=(7, self.node_len)))).astype(np.uint8)

        cdef np.ndarray[ndim=1, dtype=np.uint8_t] half_time_days = np.maximum(2, np.random.normal(loc=self.avg_halftime_days, scale=self.sigma_halftime_days, size=self.node_len)).astype(np.uint8)
        cdef np.ndarray[ndim=1, dtype=np.uint8_t] full_time_days = np.minimum(self.max_work_days, np.maximum(4, np.random.normal(loc=self.avg_work_days, scale=1, size=self.node_len))).astype(np.uint8)
        print(part_time[0])
        print(part_time[0] * half_time_days)
        print((np.uint8(1) - part_time) * full_time_days)
        cdef np.ndarray[ndim=2, dtype=np.float32_t] work_probabilities = np.tile((part_time[0] * half_time_days + (np.uint8(1) - part_time[0]) * full_time_days) / 6, (6, 1)).astype(np.float32)
        print(np.asarray(work_probabilities))
        cdef np.uint8_t[:, :] work_matrix = np.vstack((np.random.binomial(p=work_probabilities, n=1, size=(6, self.node_len)).astype(np.uint8), np.zeros(shape=self.node_len, dtype=np.uint8)))
        print(np.asarray(work_matrix))
        
        
        # when each person's sleep ends
        cdef np.ndarray[ndim=1, dtype=np.int8_t] sleep_ends = np.minimum(24, work_starting_times - ready_times).astype(np.int8)
        cdef np.int8_t[:] sleep_end_view = sleep_ends


        # when each person's sleep starts
        cdef np.ndarray[ndim=1, dtype=np.int8_t] sleep_starts = np.mod(sleep_ends - sleep_hours, 48)
        cdef np.int8_t[:] sleep_start_view = sleep_starts

        # adjusting sleep ends to make space for getting ready before work
        cdef np.ndarray[ndim=1, dtype=np.int8_t] sleep_adjustments
        sleep_adjustments = np.minimum((np.mod((work_starting_times - sleep_starts), 48) - sleep_hours - ready_times), 0).astype(np.int8)
        sleep_hours = sleep_hours + sleep_adjustments

        # getting default means of transport for each person
        cdef np.ndarray[ndim=1, dtype=np.uint8_t] transport_types = np.random.choice(a=np.arange(self.transport_probabilities.shape[0]), p=self.transport_probabilities, size=self.node_len).astype(np.uint8)
        cdef np.uint8_t[:] transport_type_view = transport_types
        

        #print(transport_types)

        cdef np.ndarray[ndim=1, dtype=np.uint8_t] school_info = 2 * np.minimum(np.maximum(np.random.normal(loc=self.school_hours_avg, scale=self.school_hours_sigma, size=self.node_len), 4), (self.school_max_end - self.school_day_start)).astype(np.uint8)
        cdef np.uint8_t[:] school_info_view = school_info
        cdef np.ndarray[ndim=1, dtype=np.uint8_t] school_sleep = (2 * np.minimum(np.maximum(np.random.normal(loc=self.sleep_avg, scale=self.sleep_sigma, size=self.node_len), 5), 10)).astype(np.uint8)
        cdef np.uint8_t[:] school_sleep_view = school_sleep 

        #print('school_info', school_info)

        cdef np.uint8_t counter = 0
        cdef FastTimeTable tmpObj
        cdef Py_ssize_t hour, day
        cdef np.uint32_t fac_id, tmp_inf, from_home_transit, from_work_transit, from_edu_transit, home_id, work_id, edu_id
        
        cdef np.uint8_t district_id, work_district_id, edu_district_id, home_district_id, transit_type, tmp_work_hours

        cdef np.uint32_t max_value = np.uint32(-1)

        


        print('Generating personal schedules...')
        start = time.time()
        cdef np.float64_t obj_start, obj_end, work_start, work_end, work_time, edu_start, edu_end, edu_time, both_start, both_end, both_time, none_start, none_end, none_time
        cdef np.float64_t obj_duration = 0
        
        cdef np.float64_t tmp_start, tmp_end, tmp_time
        
        tmp_time = 0
        work_time = 0
        edu_time = 0
        both_time = 0
        none_time = 0
        print('Made it to the loop')
        for index in range(self.node_len):
            #print('generating table for node', index)
            
            tmp_start = time.time()

            table = FastTimeTable(self.node_ids[index], age_view[index], social_view[index], self.happiness_view[index])
            

           
            table.prepare_personal_facilities(self.pers_fac_view[index, :])
            
            home_id = self.pers_fac_view[index, 0]
            edu_id = self.pers_fac_view[index, 1]
            work_id = self.pers_fac_view[index, 2]
            
            
            home_district_id = self.node_district_info_view[index, 0]
            table.set_home_district(home_district_id)

            transit_type = transport_type_view[index]
            table.set_standard_transit(transit_type)
            from_home_transit = -1
            from_work_transit = -1
            from_edu_transit = -1

            


            if work_id != max_value:
                work_district_id = self.node_district_info_view[index, 2]
                table.set_work_district(work_district_id)
                #print('Work', work_id, work_district_id, from_work_transit)

            if edu_id != max_value:
                edu_district_id = self.node_district_info_view[index, 1]
                table.set_edu_district(edu_district_id)
                #print('Edu', edu_id, edu_district_id, from_edu_transit)

            
            
            # first we create the base skeleton schedule
            tmp_time = tmp_time + (time.time() - tmp_start)
            for day in range(0, 7):
                # creating work schedules#
                
                if work_id != max_value and edu_id == max_value:
                    work_start = time.time()
                    for hour in range(sleep_view[index]):
                        table.fill_slot(day, (sleep_start_view[index] + hour + 48) % 48, 0, 0, 0, home_id, home_district_id)
                        self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=0, type=0, fac_id=home_id, day=day, hour=(sleep_start_view[index] + hour + 48) % 48, district=home_district_id)
                    if work_matrix[day, index] == 1:
                        
                        
                        
                        #print('Created sleep schedules for node', index, 'on day', day, 'for hour',  (sleep_start_view[index] + hour) % 48)
                        for hour in range(work_view[index]):
                            table.fill_slot(day, (work_start_times_view[index] + hour + 48) % 48, 1, 0, 2, work_id, work_district_id)
                            self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=0, type=2, fac_id=work_id, day=day, hour=(work_start_times_view[index] + hour + 48) % 48, district=work_district_id)
                            # table.fill_slot(day, (work_starting_times[index] + hour) % 48, 1, 0, 0, self.pers_fac_view[index, 2])

                        #print('Created Sleep and work successfully for node', index, 'on day', day)


                        if not self.transport_has_schedule_view[transit_type]:
                            fac_id = self.node_ids[index]
                            table.fill_slot(day=day, hour=(work_start_times_view[index] - 1 + 48) % 48, slot_type=2, generation_type=2, facility_type=transit_type, facility_id = fac_id, district_id = home_district_id)
                            #print('Created Transit')
                        else:
                            counter = 0
                            while from_home_transit == max_value and counter <=3:
                                from_home_transit = self.facility_manager.get_infrastructure_for_district(type=transport_type_view[index], district=home_district_id)
                                #print(from_home_transit)
                                if not self.facility_manager.check_availability(day, (work_start_times_view[index] - 1 + 48)% 48, 2, transport_type_view[index], from_home_transit, home_district_id):
                                    #print('Transit not available, WHY?')
                                    from_home_transit = max_value
                                counter += 1
                            #print(from_home_transit)
                            if from_home_transit == max_value:
                                table.fill_slot(day=day, hour=(work_start_times_view[index] - 1 + 48)% 48, slot_type=2, generation_type=3, facility_type=self.transport_probabilities.shape[0]-1, facility_id=self.node_ids[index], district_id = home_district_id)
                                
                                self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=3, type=transit_type, fac_id=fac_id, day=day, hour=(work_start_times_view[index] - 1 + 48)% 48, district=home_district_id)
                            else:
                                fac_id = from_home_transit
                                table.fill_slot(day=day, hour=(work_start_times_view[index] - 1 + 48)% 48, slot_type=2, generation_type=2, facility_type=transit_type, facility_id=fac_id, district_id=home_district_id)
                                self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=2, type=transit_type, fac_id=fac_id, day=day, hour=(work_start_times_view[index] - 1 + 48)% 48, district=home_district_id)
                            #print('Created non-car transit')

                        for hour in range(ready_time_view[index] - 1):
                            table.fill_slot(day, (work_start_times_view[index] - hour - 2 + 48) % 48, slot_type=2, generation_type=0, facility_type=0, facility_id=home_id, district_id=home_district_id)
                            self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=0, type=0, fac_id=home_id, day=day, hour=(work_start_times_view[index] - hour - 2 + 48) % 48, district=home_district_id)
                        counter = 0
                        #print('Created Ready Times')



                        if not self.transport_has_schedule_view[transit_type]:
                            fac_id = self.node_ids[index]
                            table.fill_slot(day=day, hour=(work_start_times_view[index] + work_view[index] + 48)% 48, slot_type=2, generation_type=2, facility_type=transit_type, facility_id = fac_id, district_id = work_district_id)
                            #print('Created Transit')
                        else:
                            counter = 0
                            while from_work_transit == max_value and counter <=3:
                                #print(work_district_id)
                                from_work_transit = self.facility_manager.get_infrastructure_for_district(type=transport_type_view[index], district=work_district_id)
                                if not self.facility_manager.check_availability(day, (work_start_times_view[index] + work_view[index] + 48)% 48, 2, transport_type_view[index], from_work_transit, work_district_id):
                                    from_work_transit = max_value

                                counter += 1

                            if from_work_transit == max_value:
                                table.fill_slot(day=day, hour=(work_start_times_view[index] + work_view[index] + 48)% 48, slot_type=2, generation_type=3, facility_type=self.transport_probabilities.shape[0]-1, facility_id=self.node_ids[index], district_id=work_district_id)
                                self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=3, type=transit_type, fac_id=fac_id, day=day, hour=(work_start_times_view[index] + work_view[index] + 48)% 48, district=work_district_id)
                            else:
                                fac_id = from_work_transit
                                table.fill_slot(day=day, hour=(work_start_times_view[index] + work_view[index] + 48)% 48, slot_type=2, generation_type=2, facility_type=transit_type, facility_id=fac_id, district_id=work_district_id)
                                self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=2, type=transit_type, fac_id=fac_id, day=day, hour=(work_start_times_view[index] + work_view[index] + 48)% 48, district=work_district_id)
                            #print('Created non-car transit')





                        
                            

                    
                # creating edu schedules
                    work_end = time.time()
                    work_time += work_end - work_start
                    
                elif edu_id != max_value and work_id==max_value:
                    
                    edu_start = time.time()
                    #print(school_sleep_view[index], self.school_day_start, ready_time_view[index])
                    for hour in range(school_sleep_view[index]):
                        #print((self.school_day_start - ready_time_view[index] - hour - 1) % 48)
                        #print(ready_time_view[index], hour)
                        table.fill_slot(day, (self.school_day_start - ready_time_view[index] - hour - 1 + 48) % 48, 0, 0, 0, home_id, home_district_id)
                        self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=0, type=0, fac_id=home_id, day=day, hour=(self.school_day_start - ready_time_view[index] - hour - 1 + 48) % 48, district=home_district_id)
                    
                    if day < 5:
                        #print('Created sleep schedules for node', index, 'on day', day, 'for hour',  (sleep_start_view[index] + hour) % 48)
                        for hour in range(school_info_view[index]):
                            table.fill_slot(day, (self.school_day_start + hour + 48) % 48, 1, 0, 1, edu_id, edu_district_id)
                            self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=0, type=1, fac_id=edu_id, day=day, hour=(self.school_day_start + hour + 48) % 48, district=edu_district_id)
                            # table.fill_slot(day, (work_starting_times[index] + hour) % 48, 1, 0, 0, self.pers_fac_view[index, 2])

                        #print('Created Sleep and School successfully for node', index, 'on day', day)


                        if not self.transport_has_schedule_view[transit_type]:
                            fac_id = self.node_ids[index]
                            table.fill_slot(day=day, hour=(self.school_day_start - 1 + 48)% 48, slot_type=2, generation_type=2, facility_type=transit_type, facility_id = fac_id, district_id = home_district_id)
                            #print('Created Transit')
                        else:
                            counter = 0
                            while from_home_transit == max_value and counter <=3:
                                from_home_transit = self.facility_manager.get_infrastructure_for_district(type=transport_type_view[index], district=home_district_id)
                                #print(from_home_transit)
                                if not self.facility_manager.check_availability(day, (self.school_day_start - 1 + 48)% 48, 2, transport_type_view[index], from_home_transit, home_district_id):
                                    #print('Transit not available, WHY?')
                                    from_home_transit = max_value
                                counter += 1
                            #print(from_home_transit)
                            if from_home_transit == max_value:
                                table.fill_slot(day=day, hour=(self.school_day_start - 1 + 48)% 48, slot_type=2, generation_type=3, facility_type=self.transport_probabilities.shape[0]-1, facility_id=self.node_ids[index], district_id=home_district_id)
                                self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=3, type=transit_type, fac_id=fac_id, day=day, hour=(self.school_day_start - 1 + 48)% 48, district=home_district_id)
                            else:
                                fac_id = from_home_transit
                                table.fill_slot(day=day, hour=(self.school_day_start - 1 + 48)% 48, slot_type=2, generation_type=2, facility_type=transit_type, facility_id=fac_id, district_id=home_district_id)
                                self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=2, type=transit_type, fac_id=fac_id, day=day, hour=(self.school_day_start - 1 + 48)% 48, district=home_district_id)
                            #print('Created non-car transit')

                        for hour in range(ready_time_view[index] - 1):
                            table.fill_slot(day, (self.school_day_start - hour - 2 + 48) % 48, slot_type=1, generation_type=0, facility_type=0, facility_id=home_id, district_id=home_district_id)
                            self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=0, type=0, fac_id=home_id, day=day, hour=(self.school_day_start - hour - 2 + 48) % 48, district=home_district_id)
                        counter = 0
                        #print('Created Ready Times')



                        if not self.transport_has_schedule_view[transit_type]:
                            fac_id = self.node_ids[index]
                            table.fill_slot(day=day, hour=(self.school_day_start + school_info_view[index] + 48)% 48, slot_type=2, generation_type=2, facility_type=transit_type, facility_id = fac_id, district_id=edu_district_id)
                            #print('Created Transit')
                        else:
                            counter = 0
                            while from_edu_transit == max_value and counter <=3:
                                #print(work_district_id)
                                from_edu_transit = self.facility_manager.get_infrastructure_for_district(type=transport_type_view[index], district=edu_district_id)
                                if not self.facility_manager.check_availability(day, (self.school_day_start + school_info_view[index] + 48)% 48, 2, transport_type_view[index], from_edu_transit, edu_district_id):
                                    from_edu_transit = max_value

                                counter += 1

                            if from_edu_transit == max_value:
                                table.fill_slot(day=day, hour=(self.school_day_start + school_info_view[index] + 48)% 48, slot_type=2, generation_type=3, facility_type=self.transport_probabilities.shape[0]-1, facility_id=self.node_ids[index], district_id=edu_district_id)
                                self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=3, type=transit_type, fac_id=fac_id, day=day, hour=(self.school_day_start + school_info_view[index] + 48)% 48, district=edu_district_id)
                            else:
                                fac_id = from_edu_transit
                                self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=2, type=transit_type, fac_id=fac_id, day=day, hour=(self.school_day_start + school_info_view[index] + 48)% 48, district=edu_district_id)
                                table.fill_slot(day=day, hour=(self.school_day_start + school_info_view[index] + 48)% 48, slot_type=2, generation_type=2, facility_type=transit_type, facility_id=fac_id, district_id=edu_district_id)
                        
                        
                    
                    # creating edu and work schedules
                    edu_end = time.time()
                    edu_time += edu_end - edu_start
                    
                elif work_id != max_value and edu_id != max_value:
                    both_start = time.time()
                    for hour in range(school_sleep_view[index]):
                        #print((self.school_day_start - ready_time_view[index] - hour - 1) % 48)
                        #print(ready_time_view[index], hour)
                        table.fill_slot(day, (self.school_day_start - ready_time_view[index] - hour - 1 + 48) % 48, 0, 0, 0, home_id, district_id=home_district_id)
                        self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=0, type=0, fac_id=home_id, day=day, hour=(self.school_day_start - ready_time_view[index] - hour - 1 + 48) % 48, district=home_district_id)
                    
                    if day < 5:
                        #print('Created sleep schedules for node', index, 'on day', day, 'for hour',  (sleep_start_view[index] + hour) % 48)
                        for hour in range(school_info_view[index]):
                            table.fill_slot(day, (self.school_day_start + hour + 48) % 48, 1, 0, 1, edu_id, edu_district_id)
                            self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=0, type=1, fac_id=edu_id, day=day, hour=(self.school_day_start + hour + 48) % 48, district=edu_district_id)
                            # table.fill_slot(day, (work_starting_times[index] + hour) % 48, 1, 0, 0, self.pers_fac_view[index, 2])

                        #print('Created Sleep and School successfully for node', index, 'on day', day)


                        if not self.transport_has_schedule_view[transit_type]:
                            fac_id = self.node_ids[index]
                            table.fill_slot(day=day, hour=(self.school_day_start - 1 + 48)% 48, slot_type=2, generation_type=2, facility_type=transit_type, facility_id = fac_id, district_id=home_district_id)
                            #print('Created Transit')
                        else:
                            counter = 0
                            while from_home_transit == max_value and counter <=3:
                                from_home_transit = self.facility_manager.get_infrastructure_for_district(type=transport_type_view[index], district=home_district_id)
                                #print(from_home_transit)
                                if not self.facility_manager.check_availability(day, (self.school_day_start - 1 + 48)% 48, 2, transport_type_view[index], from_home_transit, home_district_id):
                                    #print('Transit not available, WHY?')
                                    from_home_transit = max_value
                                counter += 1
                            #print(from_home_transit)
                            if from_home_transit == max_value:
                                table.fill_slot(day=day, hour=(self.school_day_start - 1 + 48)% 48, slot_type=2, generation_type=3, facility_type=self.transport_probabilities.shape[0]-1, facility_id=self.node_ids[index], district_id=home_district_id)
                                self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=3, type=transit_type, fac_id=fac_id, day=day, hour=(self.school_day_start - 1 + 48)% 48, district=home_district_id)
                            else:
                                fac_id = from_home_transit
                                table.fill_slot(day=day, hour=(self.school_day_start - 1 + 48)% 48, slot_type=2, generation_type=2, facility_type=transit_type, facility_id=fac_id, district_id=home_district_id)
                                self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=2, type=transit_type, fac_id=fac_id, day=day, hour=(self.school_day_start - 1 + 48)% 48, district=home_district_id)
                            #print('Created non-car transit')

                        for hour in range(ready_time_view[index] - 1):
                            table.fill_slot(day, (self.school_day_start - hour - 2 + 48) % 48, slot_type=2, generation_type=0, facility_type=0, facility_id=home_id, district_id=home_district_id)
                            self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=0, type=0, fac_id=home_id, day=day, hour=(self.school_day_start - hour - 2 + 48) % 48, district=home_district_id)
                        counter = 0
                        #print('Created Ready Times')


                    if work_matrix[day, index]:
                        if not self.transport_has_schedule_view[transit_type]:
                            fac_id = self.node_ids[index]
                            table.fill_slot(day=day, hour=(self.school_day_start + school_info_view[index] + 48)% 48, slot_type=2, generation_type=2, facility_type=transit_type, facility_id = fac_id, district_id=edu_district_id)
                            #print('Created Transit')
                        else:
                            counter = 0
                            while from_edu_transit == max_value and counter <=3:
                                #print(work_district_id)
                                from_edu_transit = self.facility_manager.get_infrastructure_for_district(type=transport_type_view[index], district=edu_district_id)
                                if not self.facility_manager.check_availability(day, (self.school_day_start + school_info_view[index] + 48)% 48, 2, transport_type_view[index], from_edu_transit, edu_district_id):
                                    from_edu_transit = max_value

                                counter += 1

                            if from_edu_transit == max_value:
                                table.fill_slot(day=day, hour=(self.school_day_start + school_info_view[index] + 48)% 48, slot_type=2, generation_type=3, facility_type=self.transport_probabilities.shape[0]-1, facility_id=self.node_ids[index], district_id=edu_district_id)
                                self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=3, type=transit_type, fac_id=fac_id, day=day, hour=(self.school_day_start + school_info_view[index] + 48)% 48, district=edu_district_id)
                            else:
                                fac_id = from_edu_transit
                                self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=2, type=transit_type, fac_id=fac_id, day=day, hour=(self.school_day_start + school_info_view[index] + 48)% 48, district=edu_district_id)
                                table.fill_slot(day=day, hour=(self.school_day_start + school_info_view[index] + 48)% 48, slot_type=2, generation_type=2, facility_type=transit_type, facility_id=fac_id, district_id=edu_district_id)

                        tmp_work_hours = halftime_view[index]

                        for hour in range(tmp_work_hours):
                            self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=0, type=2, fac_id=work_id, day=day, hour=(self.school_day_start + school_info_view[index] + hour + 48)% 48, district=work_district_id)
                            table.fill_slot(day=day, hour=(self.school_day_start + school_info_view[index] + hour + 48)% 48, slot_type=1, generation_type=0, facility_type=2, facility_id=work_id, district_id=work_district_id)

                    # creating uneducated and unemployed schedules
                        both_end = time.time()
                        both_time += both_end - both_start
                
                else:
                    print('Hartz IV')
                    none_start = time.time()
                    for hour in range(sleep_view[index] + ready_time_view[index]):
                        self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=0, type=0, fac_id=home_id, day=day, hour=(sleep_start_view[index] + hour) % 48, district=home_district_id) 
                        table.fill_slot(day=day, hour=(sleep_start_view[index] + hour) % 48, slot_type=0, generation_type=0, facility_type=0, facility_id=home_id, district_id=home_district_id)
                    none_end = time.time()
                    none_time += none_end - none_start
            table.fill_empty_slots()
            self.timetable_view[index] = table



            

        
            

        end = time.time()
        print('Generating personal schedules took', end-start, 'seconds', tmp_time, work_time, edu_time, both_time, none_time)
        
        #cdef FastTimeTable tmp = self.timetable_view[0]
        #tmp.print_schedule_on_day(0)





    
    cdef np.uint32_t find_district_to_node(self, Py_ssize_t node_index, Py_ssize_t facility_type, np.uint32_t[:, :] distr_view, np.uint32_t[:, :] homes):
        #print('Getting district for node', node_index)
        cdef np.uint32_t facility_id = self.pers_fac_view[node_index, facility_type]
        #print('Facility', facility_id, facility_type)


        cdef FastTimeTable tmp
        # print(facility_id)
        cdef Py_ssize_t index
        # print(homes.shape, distr_view.shape, homes.shape)
        for index in range(0, homes.shape[1]):
            if homes[0, index] == facility_id:
                # print(distr_view[facility_type, index + 1])
                # print(homes[0, index], distr_view[facility_type, index + 1], distr_view[facility_type, index])

                return distr_view[facility_type, index + 1]


        return -1
    
    cdef void put_node_in_quarantine(self, np.uint32_t node_id, np.uint8_t time_span):
        cdef FastTimeTable tmp = <FastTimeTable> self.timetable_view[node_id]
        tmp.put_in_quarantine(time_span)
        cdef Py_ssize_t day, hour
        cdef np.uint32_t[:] tmp_info
        
        for hour in range(48):
            for day in range(7):
                tmp_info = tmp.stay_home(day, hour)
                self.facility_manager.remove_node_from_facility(node_id, tmp_info[0], tmp_info[1], tmp_info[2], day, hour, tmp_info[3])
    
    cdef void infect_node(self, np.uint32_t node_id, np.uint8_t length, np.uint8_t symptoms, np.uint8_t incubation_period):
        cdef FastTimeTable tmp = <FastTimeTable> self.timetable_view[node_id]
        tmp.infect(length, symptoms, incubation_period)
    
    cdef np.uint8_t[:] get_status_update(self, np.uint32_t node_id):
        cdef FastTimeTable tmp = <FastTimeTable> self.timetable_view[node_id]
        cdef np.uint8_t[:] output = None
        output = tmp.get_status()
        return output

    cdef np.uint8_t[:] update_nodes(self):
        cdef Py_ssize_t index
        cdef FastTimeTable tmp
        cdef np.uint8_t[:] node_status = np.zeros(shape=self.node_len, dtype=np.uint8)
        for index in range(self.node_len):
            tmp = <FastTimeTable> self.timetable_view[index]
            node_status[index] = tmp.update_status()
        return node_status

    cdef np.uint8_t[:, :] get_status_for_all(self):
        cdef Py_ssize_t index
        cdef FastTimeTable tmp
        cdef np.uint8_t[:] tmp_view
        cdef np.uint8_t[:, :] node_status = np.zeros(shape=(self.node_len, 10), dtype=np.uint8)
        for index in range(self.node_len):
            tmp = <FastTimeTable> self.timetable_view[index]
            tmp_view = tmp.get_status()
            node_status[index, :] = tmp_view[:]
        return node_status

    cdef np.uint8_t schedule_appointment(self, np.uint8_t day, np.uint32_t node_one, np.uint32_t node_two, np.uint8_t generator, np.uint8_t desired_length):
        if desired_length < 3:
            return False
        cdef FastTimeTable table_one = <FastTimeTable> self.timetable_view[node_one]
        cdef FastTimeTable table_two = <FastTimeTable> self.timetable_view[node_two]
        cdef np.uint8_t[:] indices_one = table_one.get_flexible_indices_on_day(day)
        cdef np.uint8_t[:] indices_two = table_two.get_flexible_indices_on_day(day)
        cdef np.uint8_t[::1] intersection
        if indices_one.shape[0] > indices_two.shape[0]:
            intersection = np.zeros(shape=indices_two.shape[0] + 1, dtype=np.uint8)
        else:
            intersection = np.zeros(shape=indices_one.shape[0] + 1, dtype=np.uint8)
        intersection[0] = 1
        cdef np.uint8_t max_length = 0
        cdef np.uint8_t cur_length = 0
        cdef np.uint8_t max_start = 0
        cdef np.uint8_t cur_start = 0

        

        cdef Py_ssize_t index_one, index_two
        for index_one in range(indices_one.shape[0]):
            for index_two in range(indices_two.shape[0]):
                if indices_one[index_one] == indices_two[index_two]:
                    intersection[intersection[0]] = indices_one[index_one]
                    if intersection[0] == 1:
                        cur_length = 1
                        cur_start = intersection[intersection[0]]
                        max_length = 1
                        max_start = cur_start
                    elif intersection[intersection[0]] - 1 == intersection[intersection[0] - 1]:
                        cur_length += 1
                        
                        if cur_length > max_length:
                            max_length = cur_length
                            max_start = cur_start
                    elif intersection[intersection[0]] - 1 != intersection[intersection[0] - 1]:
                        cur_length = 1
                        cur_start = intersection[intersection[0]]
                    intersection[0] += 1
                    break
        #print(np.asarray(indices_one), np.asarray(indices_two))
        #print(max_length, max_start)
        #print('intersection', np.asarray(intersection))
        if intersection[0] == 1:
            return False

        if max_length > desired_length:
            max_length = desired_length
        elif max_length <= 2:
            return False

        cdef np.uint16_t district_one, district_two, district_three, district_four
        #print(max_start)
        district_one = table_one.get_district_to_hour(day, max_start)
        district_two = table_two.get_district_to_hour(day, max_start)

        if district_one != district_two:
            #print('Transit has to take place.')
            #print(node_two, day, max_start, district_two)
            self.put_node_in_transit(node_two, day, max_start, district_two)
            #print(district_two)
        
        #for index_one in range(max_length):
        #print(district_one, district_two)
        #print(max_start + max_length)
        if max_start + max_length < 48: 
            district_three = table_one.get_district_to_hour(day, max_start + max_length)
            district_four = table_two.get_district_to_hour(day, max_start + max_length)

            if district_one != district_three:
                self.put_node_in_transit(node_one, day, max_start + max_length, district_one)
            if district_two != district_four:
                self.put_node_in_transit(node_one, day, max_start + max_length, district_two)
        cdef np.uint8_t activity_count = self.facility_manager.get_activity_count()
        cdef np.ndarray p = self.activity_probabilities[node_one, :, day] * self.connection_facility
        cdef np.uint8_t activity 
        if np.sum(p) == 0:
            generator = 0
        else:
            activity = np.random.choice( p=p/np.sum(p), a=activity_count, size=1)
        
        
        #print(self.activity_probabilities[node_one, :, day], activity)

        cdef np.uint32_t facility
        cdef np.uint8_t counter = 0
        cdef Py_ssize_t hour
        cdef np.uint8_t fac_type
        cdef np.uint8_t district 
        if generator == 0:
            
            district = table_one.get_home_district() 
            facility = table_one.get_home_id()
            fac_type = 0
        else:
            facility = self.facility_manager.get_facility_for_district(activity, district_one)
            while facility == np.uint32(-1) and counter < 3:

                facility = self.facility_manager.get_facility_for_district(activity, district_one)
                if not self.facility_manager.check_availability_timespan(day=day, start=max_start + 1, end=max_start+max_length-1, generator=1, type=activity, fac_id=facility, district=district_one):
                    facility = np.uint32(-1)
                counter += 1
            fac_type = activity
        if facility == np.uint32(-1):
        
            print('No Facility found')
            for hour in range(1, max_length - 1):
                print(max_start+hour)
                table_one.fill_slot(day=day, hour=max_start+hour, slot_type=5, generation_type=3, facility_type=self.transport_probabilities.shape[0]-1, facility_id = node_one, district_id = district_one)
                table_two.fill_slot(day=day, hour=max_start+hour, slot_type=5, generation_type=3, facility_type=self.transport_probabilities.shape[0]-1, facility_id = node_two, district_id = district_one)
                self.facility_manager.add_node_to_facility(node_id=node_one, generator=3, type=self.transport_probabilities.shape[0]-1, fac_id=node_one, day=day, hour=max_start+hour, district=district_one)
                self.facility_manager.add_node_to_facility(node_id=node_two, generator=3, type=self.transport_probabilities.shape[0]-1, fac_id=node_two, day=day, hour=max_start+hour, district=district_one)
        else:
            print('Facility found')
            for hour in range(1, max_length - 1):
                table_one.fill_slot(day=day, hour=max_start+hour, slot_type=5, generation_type=generator, facility_type=fac_type, facility_id = facility, district_id = district_one)
                table_two.fill_slot(day=day, hour=max_start+hour, slot_type=5, generation_type=generator, facility_type=fac_type, facility_id = facility, district_id = district_one)
                self.facility_manager.add_node_to_facility(node_id=node_one, generator=generator, type=fac_type, fac_id=facility, day=day, hour=max_start+hour, district=district_one)
                self.facility_manager.add_node_to_facility(node_id=node_two, generator=generator, type=fac_type, fac_id=facility, day=day, hour=max_start+hour, district=district_one)
        #table_one.print_strings_on_day(day)
        #table_two.print_strings_on_day(day)
        return True

    cdef np.uint8_t put_node_in_transit(self, np.uint32_t node_id, np.uint8_t day, np.uint8_t hour, np.uint8_t district):
        
        cdef FastTimeTable tmpObj = <FastTimeTable> self.timetable_view[node_id]
        cdef np.uint8_t transit_type = tmpObj.get_standard_transit()
        #print(transit_type)
        cdef np.uint32_t tmp_id 
        cdef np.uint8_t counter = 0
        if not self.transport_has_schedule_view[transit_type]:
            tmpObj.fill_slot(day, hour, 2, 2, transit_type, node_id, district)
            return True
        else:
            tmp_id = self.facility_manager.get_infrastructure_for_district(transit_type, district)
            while tmp_id == np.uint32(-1) and counter<3:
                tmp_id = self.facility_manager.get_infrastructure_for_district(transit_type, district)
                counter += 1
            if tmp_id == np.uint32(-1):
                tmpObj.fill_slot(day, hour, 2, 3, self.transport_probabilities.shape[0]-1, node_id, district)
                self.facility_manager.add_node_to_facility(node_id, 3, transit_type, node_id, day, hour, district)
            else:
                tmpObj.fill_slot(day, hour, 2, 2, transit_type, node_id, district)
                self.facility_manager.add_node_to_facility(node_id, 2, transit_type, node_id, day, hour, district)
            return True