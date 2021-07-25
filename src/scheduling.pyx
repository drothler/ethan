#cython: language_level=3, wraparound=False, boundscheck=False, initializedcheck=False, nonecheck=False, cdivision=True
cimport scheduling
import numpy as np
cimport numpy as np
from helper cimport contains, get_binary_event
from helper import get_words_from_string, get_ints_from_string, get_floats_from_string, get_bools_from_string
import sys
import time
import csv



cdef class FastFacilityTable:


    def __init__(self, np.uint64_t unique_id, str facility_name, np.uint64_t capacity, np.uint32_t district_id):
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

    cdef int get_id(self):
        return self.unique_id


    cdef np.uint8_t add_node_at_time(self, Py_ssize_t day, Py_ssize_t hour, int id):
        cdef Py_ssize_t fill_count
        fill_count = self.schedule_view[hour, day, self.capacity]

        # print(fill_count)
        if fill_count < self.capacity:
            self.schedule_view[hour, day, fill_count] = id
            self.schedule_view[hour, day, self.capacity] += 1
            return True
        else:
            return False

    cdef np.uint8_t remove_node_from_time(self, Py_ssize_t day, Py_ssize_t hour, int id):
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

    cdef np.uint64_t get_capacity(self):
        return self.capacity

    cdef int get_available_capacity(self, int day, int hour):
        return self.capacity - self.schedule[day, hour, self.capacity]

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


    def __init__(self, np.uint32_t node_id, np.uint8_t age, float32_t introvertedness, float32_t happiness):


        self.id = node_id
        self.age = age

        # 0 - healthy, 1 - infected, 2 - cured, 3 - immune
        self.status = 0

        self.days_since_infection = -1
        self.positive_test = False
        self.shows_symptoms = False
        self.is_quarantined = False

        self.introvertedness = introvertedness

        self.sociability = introvertedness * age
        self.happiness = happiness


        self.schedule = np.full([48, 7, 4], fill_value=-1, dtype=np.uint64)
        self.schedule_view = self.schedule

        self.transit_candidates = np.ndarray(shape=(3, 3), dtype=np.uint64)
        self.transit_candidate_view = self.transit_candidates

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







    cdef fill_slot(self, Py_ssize_t day, Py_ssize_t hour, np.uint64_t slot_type, np.uint64_t generation_type, np.uint64_t facility_type, np.uint64_t facility_id):
        if self.schedule_view[hour, day, 0] == 0:
            raise ValueError('Can\'t overwrite fixed time slot on day', day, 'and hour', hour)
        elif self.schedule_view[hour, day, 0] == 1:
            raise ValueError('Can\'t overwrite routine time slot on day', day, 'and hour', hour, '\nFirst delete routine!')
        elif self.schedule_view[hour, day, 0] == 2:
            raise ValueError('Can\'t overwrite transit time slot on day', day, 'and hour', hour, '\nFirst delete transit!')
        else:
            self.schedule_view[hour, day, 0] = slot_type
            self.schedule_view[hour, day, 1] = generation_type
            self.schedule_view[hour, day, 2] = facility_type
            self.schedule_view[hour, day, 3] = facility_id

    cdef set_home_district(self, np.uint16_t district_id):
        self.home_district = district_id
    cdef set_edu_district(self, np.uint16_t district_id):
        self.edu_district = district_id
    cdef set_work_district(self, np.uint16_t district_id):
        self.work_district = district_id

    cdef void empty_slot(self, Py_ssize_t day, Py_ssize_t hour):
        if self.schedule_view[hour, day, 0] == 0:
            raise ValueError('Can\'t overwrite fixed time slot on day', day, 'and hour', hour)
        self.schedule_view[hour, day, :] = -1

    cdef np.uint64_t[:] get_flexible_indices_on_day(self, Py_ssize_t day):
        cdef Py_ssize_t index
        cdef np.uint8_t length = 0
        for index in range(self.schedule_view.shape[0]):
            if self.schedule_view[index, day, 0] == 3:
                length += 1
        length = 0
        cdef np.ndarray[ndim=1, dtype=np.uint64_t] indices = np.ndarray(shape=length, dtype=np.uint64)
        for index in range(self.schedule_view.shape[0]):
            if self.schedule_view[index, day, 0] == 3:
                indices[length] = index
            length += 1
        return indices

    # reads personal facilities from attribute_view and assigns them accordingly to home, edu and work

    cdef void prepare_personal_facilities(self, np.uint64_t[:] attribute_view):
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
        if self.work_facility != -1:
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
        print(np.asarray(self.schedule_view[:, day, 3]).reshape(1, 48))

    cdef np.uint64_t get_work_id(self):
        return self.work_facility

    cdef np.uint64_t get_edu_id(self):
        return self.edu_facility

    cdef np.uint64_t get_home_id(self):
        return self.home_facility

    cdef np.uint8_t is_quarantined(self):
        return self.is_quarantined



cdef class FacilityManager:

    def __init__(self, dict facilities, dict infrastructure, dict personals, int node_count):
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
        self.general_pool = np.zeros(shape=(node_count), dtype=bool)
        self.district_count = <int> np.maximum(1, np.float32(self.infrastructure_cfg['districts_per_million']) * self.million_ratio)

        self.initialize_facilities()
        self.initialize_infrastructures()
        self.initialize_infrastructure_by_district()

    cdef initialize_infrastructure_by_district(self):
        cdef np.ndarray[ndim=4, dtype=np.uint64_t] inf_by_dis

        cdef FastFacilityTable tmpObj
        cdef Py_ssize_t inf_type, index, max_distr
        cdef Py_ssize_t shape, len, tmp_district
        shape = self.infrastructure.shape[0]
        len = self.infrastructure.shape[1]
        cdef np.ndarray[ndim=2, dtype=np.uint64_t] district_counts = np.zeros(shape=(shape, self.district_count), dtype=np.uint64)
        cdef np.uint64_t[:, :] district_view = district_counts

        for inf_type in range(shape):
            for index in range(len):
                tmpObj = self.infrastructure[inf_type][index]
                if tmpObj is None:
                    break
                else:
                    tmp_district = tmpObj.get_district()
                    district_view[inf_type, tmp_district] += 1

        max_distr = np.max(district_counts)

        inf_by_dis = np.ndarray(shape=(self.district_count, shape, max_distr + 1, 2), dtype=np.uint64)

        cdef Py_ssize_t fill_amount
        cdef np.uint64_t[:, :, :, :] inf_view = inf_by_dis
        inf_view[:, :, 0, 0] = 1
        inf_view[:, :, 0, 1] = 1
        for inf_type in range(shape):
            for index in range(len):
                tmpObj = self.infrastructure[inf_type][index]
                if tmpObj is None:
                    break
                else:
                    tmp_district = tmpObj.get_district()
                    fill_amount = inf_view[tmp_district, inf_type, 0, 0]
                    inf_view[tmp_district, inf_type, fill_amount, 0] = inf_type
                    inf_view[tmp_district, inf_type, fill_amount, 1] = index
                    inf_view[tmp_district, inf_type, 0, 0] += 1

        self.infrastructure_by_district = inf_by_dis
        self.infrastructure_by_dis_view = self.infrastructure_by_district
        # print(np.asarray(inf_view[:, 2, :, 1]))

    cdef initialize_personal_facilities(self, np.uint32_t[:, :, :, :] personal_facilities):
        print('Generating graph-defined facilities from network...')
        cdef float start = time.time()

        cdef np.ndarray pers_spaces = np.array(get_bools_from_string(self.personal_cfg['personal_space']), dtype=bool)
        cdef np.uint8_t[:] pers_space_view = pers_spaces

        cdef np.ndarray pers_capacities = np.array(get_ints_from_string(self.personal_cfg['capacity']), dtype=int)
        cdef int[:] pers_capacity_view = pers_capacities

        cdef np.ndarray capacities = np.ndarray([self.district_count, personal_facilities.shape[1], personal_facilities.shape[2]], dtype=int)
        cdef int[:, :, :] capacity_view = capacities

        self.personal_facilities = np.ndarray([self.district_count, personal_facilities.shape[1], personal_facilities.shape[2]], dtype=object)

        cdef Py_ssize_t district_index, type_index, facility_index
        cdef FastFacilityTable tmp
        for district_index in range(self.district_count):
            for type_index in range(personal_facilities.shape[1]):
                for facility_index in range(personal_facilities[district_index, type_index, 0, 0]):
                    capacity_view[district_index, type_index, facility_index] = personal_facilities[district_index, type_index, facility_index + 1, 1] * pers_capacity_view[type_index]
                    tmp = FastFacilityTable(facility_name='test', unique_id=personal_facilities[district_index, type_index, facility_index + 1, 0], capacity=capacity_view[district_index, type_index, facility_index], district_id=district_index)
                    self.personal_facilities[district_index][type_index][facility_index] = tmp
                    if personal_facilities[district_index, type_index, facility_index + 1, 0] == 506502:
                        print('DISTRICT TEST:', district_index, type_index, facility_index)


        self.pers_view = self.personal_facilities
        cdef float end = time.time()
        print('Initializing graph-defined facilities took', end-start, 'seconds')

    cdef initialize_facilities(self):
        print('Generating user-defined facilities from config file...')
        cdef float start = time.time()
        cdef list facility_types = get_words_from_string(self.facility_cfg['autogenerated_facilities'])

        cdef np.ndarray capacity_avgs = np.array(get_ints_from_string(self.facility_cfg['capacity_avg']), dtype=int)

        cdef np.ndarray capacity_sigmas = np.array(get_floats_from_string(self.facility_cfg['capacity_sigma']), dtype=np.float32)

        cdef np.ndarray activity_types = np.array(get_ints_from_string(self.facility_cfg['activity_type']), dtype=int)

        cdef np.ndarray facilities_per_million = np.array(get_ints_from_string(self.facility_cfg['facility_per_million']), dtype=int)

        cdef Py_ssize_t list_index
        cdef Py_ssize_t list_len = len(facility_types)

        cdef np.ndarray instances = facilities_per_million * self.million_ratio
        cdef np.ndarray seperate_pools = np.array(get_bools_from_string(self.facility_cfg['seperate_pools']), dtype=bool)

        instances = np.multiply(instances, seperate_pools)
        # cdef np.ndarray corrected_instances =
        self.facilities = np.ndarray([list_len, int(np.max(facilities_per_million))], dtype=object)
        for list_index in range(list_len):
            self.initialize_facility(list_index, facility_types[list_index], capacity_avgs[list_index], capacity_sigmas[list_index], <int>instances[list_index], self.district_count)
        cdef float end = time.time()
        self.fac_view = self.facilities
        print('Initializing user-defined facilities took', end-start, 'seconds')


    cdef void initialize_facility(self, int row,  str name, int capacity_avg, float32_t capacity_sigma, int instances, int district_count):

        cdef np.ndarray capacity = np.maximum(np.random.normal(loc=capacity_avg, scale=capacity_sigma, size=instances), 0)
        cdef np.ndarray district = np.random.randint(low=0, high=district_count, size=instances)
        cdef np.float64_t[:] capacities = capacity
        cdef int[:] districts = district
        cdef int facility_id = 0
        cdef FastFacilityTable cur_facility
        for facility_id in range(instances):

            cur_facility = FastFacilityTable(unique_id=facility_id, facility_name=name, capacity=<int>capacities[facility_id], district_id=districts[facility_id])
            self.facilities[row][facility_id] = cur_facility


    cpdef int[:, :, :, :] return_facilities_by_district(self):
        cdef Py_ssize_t index, facility_id, max_district, activity_count, cur_index
        cdef int tmp_facility_id
        cdef np.ndarray facility_districts
        activity_count = self.facilities.shape[0]
        cdef np.ndarray district_counter = np.zeros([self.district_count, activity_count], dtype=int)
        cdef int[:, :] district_view = district_counter
        cdef Py_ssize_t tmp_district_id, tmp_activity_id
        cdef FastFacilityTable tmpTable

        for index in range(activity_count):
            for facility_id in range(self.facilities.shape[1]):
                if self.facilities[index][facility_id] is not None:
                    tmpTable = self.facilities[index][facility_id]
                    tmp_district_id = tmpTable.get_district()

                    district_view[tmp_district_id, index] += 1
        # print(district_counter)
        max_district = np.max(district_counter)
        facility_districts = np.full(shape=(self.district_count, activity_count, max_district + 1, 2), fill_value=-1, dtype=int)
        cdef int[:, :, :, :] view = facility_districts
        view[:, :, 0, 0] = 0
        view[:, :, 0, 1] = 0
        index = 0
        facility_id = 0
        for index in range(activity_count):
            for facility_id in range(self.facilities.shape[1]):
                if self.facilities[index, facility_id] is not None:
                    tmpTable = self.facilities[index][facility_id]
                    tmp_district_id = tmpTable.get_district()
                    tmp_facility_id = tmpTable.get_id()
                    view[tmp_district_id, index, view[tmp_district_id, index, 0, 0] + 1, 0] = index
                    view[tmp_district_id, index, view[tmp_district_id, index, 0, 0] + 1, 1] = facility_id
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

        cdef int max_instance = <int>np.max(np.multiply(needs_schedule, instances))
        # print(np.multiply(needs_schedule, instances))


        self.infrastructure = np.ndarray([inf_len, max_instance], dtype=object)

        for index in range(inf_len):
            if needs_schedule[index]:
                self.initialize_infrastructure(index, infrastructure_types[index], capacities[index], instances[index])

    cdef initialize_infrastructure(self, Py_ssize_t row, str name, int capacity_avg, int instances):
        cdef Py_ssize_t index

        cdef np.ndarray districts = np.random.randint(low=0, high=self.district_count, size=instances)
        cdef int[:] district_view = districts
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

    cdef np.uint64_t get_facility_for_district(self, Py_ssize_t type, Py_ssize_t district):
        cdef np.uint64_t random_index = self.fac_dis_view[district, type, 0, 1]
        self.fac_dis_view[district, type, 0, 1] = (self.fac_dis_view[district, type, 0, 1] % self.fac_dis_view[district, type, 0, 0]) + 1
        cdef np.uint64_t index = self.fac_dis_view[district, type, random_index, 1]
        cdef np.uint64_t inf_id = self.fac_view[type, index]
        self.fac_dis_view[district, type, 0, 1] = (self.fac_dis_view[district, type, 0, 1] + 1)

    cdef np.uint64_t get_infrastructure_for_district(self, Py_ssize_t type, Py_ssize_t district):
        cdef np.uint64_t random_index = self.fac_dis_view[district, type, 0, 1]
        self.fac_dis_view[district, type, 0, 1] = (self.fac_dis_view[district, type, 0, 1] % self.fac_dis_view[district, type, 0, 0]) + 1
        cdef np.uint64_t index = self.fac_dis_view[district, type, random_index, 1]
        cdef np.uint64_t inf_id = self.fac_view[type, index]
        self.fac_dis_view[district, type, 0, 1] = (self.fac_dis_view[district, type, 0, 1] + 1)


    cdef np.uint8_t add_node_to_facility(self, np.uint32_t node_id, Py_ssize_t generator, Py_ssize_t type, np.uint64_t fac_id, Py_ssize_t day, Py_ssize_t hour, Py_ssize_t district):
        cdef Py_ssize_t index
        # 0 means personal facilities (school, work, edu)

        # 1 means config facilities

        # 2 means infrastructure

        cdef np.uint64_t tmp_type, tmp_index, tmp_id
        cdef FastFacilityTable tmpObj
        cdef np.uint8_t return_value = False

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

            for index in range(self.pers_view[district, type].shape[0]):
                tmpObj = self.pers_view[district, type, index]
                tmp_id = tmpObj.get_id()
                if tmp_id == fac_id:
                    #return_value = tmpObj.add_node_at_time(day, hour, node_id)
                    #self.pers_view[district, type, index] = tmpObj
                    return tmpObj.add_node_at_time(day, hour, node_id)

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
                        return_value = tmpObj.add_node_at_time(day, hour, node_id)
                        self.pers_view[district, type, index] = tmpObj
                        return return_value
                else:
                    break
            return False
        else:
            for index in range(self.infrastructure_by_dis_view[district, type, 0, 0]):
                tmp_type = self.infrastructure_by_dis_view[district, type, index + 1, 0]
                tmp_index = self.infrastructure_by_dis_view[district, type, index + 1, 1]
                tmpObj = self.infrastructure_view[tmp_type, tmp_index]
                if tmpObj is not None:
                    tmp_id = tmpObj.get_id()
                    if tmp_id == fac_id:
                        return_value = tmpObj.add_node_at_time(day, hour, node_id)
                        self.pers_view[district, type, index] = tmpObj
                        return return_value
                else:
                    break
            return False


    cdef np.uint8_t remove_node_from_facility(self, np.uint32_t node_id, Py_ssize_t generator, Py_ssize_t type, np.uint64_t fac_id, Py_ssize_t day, Py_ssize_t hour, Py_ssize_t district):
        cdef Py_ssize_t index
        # 0 means personal facilities (school, work, edu)

        # 1 means config facilities

        # 2 means infrastructure

        cdef np.uint64_t tmp_type, tmp_index, tmp_id
        cdef FastFacilityTable tmpObj

        cdef np.uint8_t return_value = False
        if generator == 0:

            for index in range(self.pers_view[district, type].shape[0]):

                tmpObj = self.pers_view[district, type, index]
                if tmpObj is not None:
                    tmp_id = tmpObj.get_id()
                    if tmp_id == fac_id:
                        return_value = tmpObj.remove_node_from_time(node_id, day, hour)
                        self.pers_view[district, type, index] = tmpObj
                        return return_value

                else:
                    break
            print('This shouldnt happen lol')
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
                        self.pers_view[district, type, index] = tmpObj
                        return return_value
                else:
                    break
            print('This should also not happen lol')
            return False
        else:
            for index in range(self.infrastructure_by_dis_view[district, type, 0, 0]):
                tmp_type = self.infrastructure_by_dis_view[district, type, index + 1, 0]
                tmp_index = self.infrastructure_by_dis_view[district, type, index + 1, 1]
                tmpObj = self.infrastructure_view[tmp_type, tmp_index]
                if tmpObj is not None:
                    tmp_id = tmpObj.get_id()
                    if tmp_id == fac_id:
                        return_value = tmpObj.remove_node_from_time(node_id, day, hour)
                        self.pers_view[district, type, index] = tmpObj
                        return return_value
                else:
                    break
            print('why does this happen? You have only yourself to blame')
            return False



cdef class HumanManager:

    def __init__(self, np.uint32_t[:] node_ids, np.uint32_t[:, :, :] connections, np.ndarray personal_facilities, dict legal_dict, dict population_dict, dict facilities, dict infrastructure, dict personals, dict simulation_dict, int districts):
        self.node_ids = node_ids
        self.connections = connections
        self.personal_facilities = personal_facilities
        self.pers_fac_view = self.personal_facilities
        self.node_len = node_ids.shape[0]

        # initializing an instance of facility manager automatically creates all required config facilities
        self.facility_manager = FacilityManager(facilities, infrastructure, personals, self.node_len)
        # Auto Facilities - Graph Facilities - Node Time Tables - Fill Node Tables
        #     YES                 NO                 NO                NO

        self.timetables = np.ndarray([self.node_len], dtype=object)
        self.timetable_view = self.timetables

        # filler array, i want to feed the data loader age pyramid data in the future
        self.ages = np.minimum(np.maximum(np.random.normal(size=self.node_len, loc=45, scale=10), 100), 0)
        self.age_view = self.ages
        self.introvertedness = np.minimum(np.maximum(np.random.normal(size=self.node_len, loc=0.5, scale=0.25), 1), 0)
        self.intro_view = self.introvertedness
        self.happiness = np.random.randint(low=0, high=10, size=self.node_len)
        self.happiness_view = self.happiness

        self.work_info = np.ndarray([self.node_len, 3], dtype=int)
        self.work_view = self.work_info
        self.edu_info = np.ndarray([self.node_len, 3], dtype=int)
        self.edu_view = self.edu_info

        self.district_count = districts
        self.node_district_info = np.full(shape=(self.node_len, 3), fill_value=-1, dtype=np.uint16)
        self.node_district_info_view = self.node_district_info

        self.max_work_hours = int(legal_dict['max_work_hours'])
        self.work_avg = int(population_dict['average_fulltime_work'])
        self.work_sigma = int(population_dict['sigma_fulltime_work'])
        self.work_starts_density = np.array(get_floats_from_string(population_dict['work_start_times_density']))
        self.work_starts = np.array(get_ints_from_string(population_dict['work_start_times']))
        self.work_start_view = self.work_starts

        self.sleep_avg = int(simulation_dict['sleep_hours_default'])
        self.sleep_sigma = int(simulation_dict['sleep_hours_sigma'])

        self.transport_probabilities = np.array(get_floats_from_string(infrastructure['population_percentage']))
        self.transport_probabilities = np.append(self.transport_probabilities, 1 - np.sum(self.transport_probabilities))
        self.transport_probabilities = self.transport_probabilities / np.sum(self.transport_probabilities)
        # Calling this creates all necessary
        self.initialize_work_info()
        # Auto Facilities - Graph Facilities - Node Time Tables - Fill Node Tables
        #     YES                 YES                 NO                NO

    cdef initialize_work_info(self):
        print('Initializing Node Information (districts, work/home/edu places etc)')
        cdef Py_ssize_t node_index, type_index, facility_index
        cdef Py_ssize_t work_len = 0
        cdef Py_ssize_t work_avg, work_sigma
        cdef np.ndarray[ndim=2, dtype=np.float64_t] work_length = np.minimum(np.maximum(np.random.normal(loc=self.work_avg, scale=self.work_sigma, size=(self.node_len, 7)), 1), self.max_work_hours)
        cdef np.float64_t[:, :] work_length_view = work_length


        cdef np.ndarray work_ids, work_counts, edu_ids, edu_counts, home_ids, home_counts
        work_ids, work_counts = np.unique(self.personal_facilities[:, 2], return_counts=True)
        home_ids, home_counts = np.unique(self.personal_facilities[:, 0], return_counts=True)
        edu_ids, edu_counts = np.unique(self.personal_facilities[:, 1], return_counts=True)

        cdef np.ndarray[ndim=2, dtype=np.uint64_t] work_places, edu_places, home_places
        cdef np.ndarray[ndim=3, dtype=np.uint64_t] places
        work_places = np.stack((work_ids, work_counts)).astype(np.uint64)
        edu_places = np.stack((edu_ids, edu_counts)).astype(np.uint64)
        home_places = np.stack((home_ids, home_counts)).astype(np.uint64)

        cdef Py_ssize_t max_dim = np.max(np.array([work_places.shape[1], home_places.shape[1], edu_places.shape[1]], dtype=np.uint64))

        places = np.ndarray(shape=(3, max_dim, 2), dtype=np.uint64)
        cdef np.uint64_t[:, :, :] place_view = places

        for node_index in range(home_places.shape[1]):
            place_view[0, node_index, 0] = home_places[0, node_index]
            place_view[0, node_index, 1] = home_places[1, node_index]

        for node_index in range(edu_places.shape[1]):
            place_view[1, node_index, 0] = edu_places[0, node_index]
            place_view[1, node_index, 1] = edu_places[1, node_index]

        for node_index in range(work_places.shape[1]):
            place_view[2, node_index, 0] = work_places[0, node_index]
            place_view[2, node_index, 1] = work_places[1, node_index]

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

        cdef np.uint64_t facility_id
        for node_index in range(self.node_len):
            self.node_district_info_view[node_index, 0] = self.find_district_to_node(node_index, 0, distr_view, home_places)



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

    cpdef initialize_node_info(self, np.ndarray age_distr_array):

        # creating ages based on distribution
        cdef np.ndarray[ndim=1, dtype=np.uint8_t] age_array = np.random.choice(a=np.arange(age_distr_array.shape[0]), size=self.node_len, p=age_distr_array).astype(np.uint8)
        cdef np.uint8_t[:] age_view = age_array

        # creating social scores
        cdef np.ndarray[ndim=1, dtype=np.float32_t] social_scores = np.minimum(np.maximum(np.random.normal(loc=5, scale=2.5, size=self.node_len), 0), 10).astype(np.float32)
        cdef np.float32_t[:] social_view = social_scores

        # creating happiness factor among population
        cdef np.ndarray[ndim=1, dtype=np.float32_t] happiness = np.minimum(np.maximum(np.random.normal(loc=5, scale=2.5, size=self.node_len), 0), 10).astype(np.float32)


        print('Generating personal schedules...')
        start = time.time()


        cdef Py_ssize_t index
        cdef FastTimeTable table

        # how long does each person sleep
        cdef np.ndarray[ndim=1, dtype=np.int8_t] sleep_hours = (2 * np.minimum(np.maximum(np.random.normal(loc=self.sleep_avg, scale=self.sleep_sigma, size=self.node_len), 5), 10)).astype(np.int8)
        cdef np.int8_t[:] sleep_view = sleep_hours

        # how long would each person work full time (from 3 to max_work_hours, normal distribution)
        cdef np.ndarray[ndim=1, dtype=np.int8_t] work_hours = (2 * np.minimum(np.maximum(np.random.normal(loc=self.work_avg, scale=self.work_sigma, size=self.node_len), 3), self.max_work_hours)).astype(np.int8)
        cdef np.int8_t[:] work_view = work_hours




        # when does each person start working each day
        cdef np.ndarray[ndim=1, dtype=np.int8_t] work_starting_times = (2 * np.random.choice(a=self.work_starts, p=self.work_starts_density, size=self.node_len)).astype(np.int8)
        cdef np.int8_t[:] work_start_times_view = work_starting_times

        # how long it take each person to get ready for work (from 60 to 90 minutes)
        cdef np.ndarray[ndim=1, dtype=np.int8_t] ready_times = (np.random.randint(low=2, high=3, size=self.node_len)).astype(np.int8)
        cdef np.int8_t[:] ready_time_view = ready_times




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








        cdef FastTimeTable tmpObj
        cdef Py_ssize_t hour, day
        cdef np.uint64_t fac_id
        for index in range(self.node_len):
            table = FastTimeTable(node_id=self.node_ids[index], age=age_view[index], introvertedness=social_view[index], happiness=happiness[index])
            table.prepare_personal_facilities(self.pers_fac_view[index, :])

            # first we create the base skeleton schedule

            for day in range(0, 7):
                # creating work schedules
                if table.get_work_id() != -1 and table.get_edu_id() == -1:

                    for hour in range(sleep_view[index]):
                        fac_id = self.pers_fac_view[index, 0]
                        table.fill_slot(day, (sleep_start_view[index] + hour) % 48, 1, 0, 0, fac_id)

                        self.facility_manager.add_node_to_facility(node_id=self.node_ids[index], generator=0, type=0, fac_id=fac_id, day=day, hour=(sleep_start_view[index] + hour) % 48, district=self.node_district_info[index, 0])
                    # print('Created sleep schedules for node', index, 'on day', day, 'for hour',  (sleep_start_view[index] + hour) % 48)
                    for hour in range(work_view[index]):
                        pass
                        # table.fill_slot(day, (work_starting_times[index] + hour) % 48, 1, 0, 0, self.pers_fac_view[index, 2])






                # creating edu schedules
                elif table.get_work_id() == -1 and table.get_edu_id() != -1:
                    pass

                # creating edu and work schedules

                # creating uneducated and unemployed schedules





            #table.print_schedule()
            self.timetable_view[index] = table

        end = time.time()
        print('Generating personal schedules took', end-start, 'seconds')
        cdef FastTimeTable tmp = self.timetable_view[0]
        tmp.print_schedule_on_day(0)





    cdef np.uint32_t find_district_to_node(self, Py_ssize_t node_index, Py_ssize_t facility_type, np.uint32_t[:, :] distr_view, np.uint64_t[:, :] homes):
        #print('Getting district for node', node_index)
        cdef np.uint64_t facility_id = self.pers_fac_view[node_index][facility_type]
        #print('Facility', facility_id, facility_type)


        cdef FastTimeTable tmp
        # print(facility_id)
        cdef Py_ssize_t index
        # print(homes.shape, distr_view.shape, homes.shape)
        for index in range(0, homes.shape[1]):
            if homes[0, index] == facility_id:
                #print(distr_view[facility_type, index + 1])
                # print(homes[0, index], distr_view[facility_type, index + 1], distr_view[facility_type, index])

                return distr_view[facility_type, index + 1]


        return -1