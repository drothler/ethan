#cython: language_level=3, wraparound=True, boundscheck=True, initializedcheck=True, nonecheck=True, cdivision=True


from scheduling cimport HumanManager, FastFacilityTable
from data import DataLoader
import numpy as np
cimport numpy as np
import configparser
from pympler import asizeof
import gc
from helper import get_bools_from_string, get_ints_from_string, get_floats_from_string
from libc.stdlib cimport rand, RAND_MAX, srand

cdef class FastSimulation:

    def __init__(self, np.uint8_t is_training, int city_size, np.uint8_t print_updates, np.ndarray node_np, np.ndarray info_np, np.ndarray info_matrix, dict legal_dict, dict population_dict, dict facilities, dict infrastructure, dict personals, dict simulation_dict, dict connection_dict):
        self.training = is_training
        self.facility_dict = facilities
        self.infrastructure_dict = infrastructure
        self.personal_dict = personals
        self.legal_dict = legal_dict
        self.population_dict = population_dict
        self.connection_dict = connection_dict
        self.simulation_dict = simulation_dict
        #self.human_manager = HumanManager()
        self.type_dict = {
            'friend': 1,
            'household': 2,
            'colleague': 3,
            'partner': 4,
            'family': 5
        }
        
        self.node_np = node_np
        self.info_np = info_np
        self.info_matrix = info_matrix

        self.day = 0
        
        
        self.human_manager = HumanManager(np.array(range(0, self.node_np.shape[0]), dtype=np.uint32), self.node_np, self.info_np, self.legal_dict, self.population_dict, self.facility_dict, self.infrastructure_dict, self.personal_dict, self.simulation_dict, 2 + np.uint32((int(self.infrastructure_dict['districts_per_million']) / 1e6) * self.node_np.shape[0]))
        print('Created human manager')
        self.human_manager.initialize_node_info(np.full(fill_value=0.01, dtype=np.float32, shape=100))
        self.human_manager.initialize_flexible_slots()
        srand(0)
        for index in range(100):
            self.human_manager.schedule_appointment(3, 2* index, 2* index +1, 0, 4)
        self.create_connected()
        




    cdef void create_connected(self):
        cdef Py_ssize_t node 
        cdef np.ndarray frequency_avg = np.array(get_ints_from_string(self.connection_dict['weekly_avg_frequency']), dtype=np.uint8)
        cdef np.ndarray frequency_sigma = np.array(get_ints_from_string(self.connection_dict['weekly_sigma_frequency']), dtype=np.uint8)
        
        cdef np.ndarray[ndim=2, dtype=np.uint8_t] enc_counts = np.minimum(np.maximum(np.random.normal(loc=frequency_avg, scale=frequency_sigma, size=(self.node_np.shape[0], 5)), 1), 7).astype(np.uint8)
        print(enc_counts)
        cdef np.uint8_t[:, :] enc_count_view = enc_counts
        cdef np.uint64_t total_encounters = np.sum(enc_counts)
        self.connected_encounter_ids = np.ndarray(shape=(total_encounters, 2), dtype=np.uint32)
        self.connected_encounters_facilities = np.ndarray(shape=total_encounters, dtype=np.uint32)
        self.connected_encounters_meta = np.ndarray(shape=(total_encounters, 4), dtype=np.uint8)
        self.connected_encounters_intensity = np.ndarray(shape=total_encounters, dtype=np.uint8)
        self.connected_encounters_duration = np.ndarray(shape=total_encounters, dtype=np.uint16)
        
        cdef np.uint8_t[:] intensity_avg = np.array(get_ints_from_string(self.connection_dict['intensity_avg']), dtype=np.uint8)
        cdef np.uint8_t[:] intensity_sigma = np.array(get_ints_from_string(self.connection_dict['intensity_sigma']), dtype=np.uint8)
        cdef np.uint16_t[:] duration_avg = np.array(get_ints_from_string(self.connection_dict['duration_avg']), dtype=np.uint16)
        cdef np.uint16_t[:] duration_sigma = np.array(get_ints_from_string(self.connection_dict['duration_sigma']), dtype=np.uint16)
        print(total_encounters)
        for node in range(self.node_np.shape[0]):
            self.create_connected_encounters_for_node(node, enc_count_view[node], intensity_avg, intensity_sigma, duration_avg, duration_sigma)
        

    cdef void create_connected_encounters_for_node(self, Py_ssize_t node, np.uint8_t[:] count, np.uint8_t[:] intensity_avg,  np.uint8_t[:] intensity_sigma, np.uint16_t[:] duration_avg, np.uint16_t[:] duration_sigma):
        cdef Py_ssize_t connection_type, encounter_index
        cdef np.uint8_t[:] intensities
        cdef np.uint16_t[:] durations
        cdef np.uint32_t[:] potential_nodes 
        cdef np.uint8_t random_index, generator
        cdef np.uint8_t hour_duration 
        #print(np.asarray(self.node_np[node, :, :]))
        for connection_type in range(count.shape[0]):
            if self.info_matrix[node, connection_type] and count[connection_type] > 0:
                intensities = np.random.normal(loc=intensity_avg[connection_type], scale=intensity_sigma[connection_type], size=count[connection_type]).astype(np.uint8)
                durations = np.random.normal(loc=duration_avg[connection_type], scale=duration_sigma[connection_type], size=count[connection_type]).astype(np.uint16)
                #print(np.asarray(intensities), np.asarray(durations))
                potential_nodes = self.get_nodes_to_type(node, connection_type)
                for encounter_index in range(count[connection_type]):
                    
                    random_index = np.uint8((rand()* 1.0/(RAND_MAX * 1.0))*potential_nodes.shape[0])
                    generator = np.uint8((rand()* 1.0/(RAND_MAX * 1.0))*2)
                    hour_duration = durations[encounter_index] / 30
                    print('APPOINTMENT STATUS:', self.human_manager.schedule_appointment(self.day, node, potential_nodes[random_index], generator, hour_duration))
                    #print(generator)
                    #print(random_index, potential_nodes[random_index])
                #print(node, 'Potential nodes for type', connection_type, np.asarray(potential_nodes))


    cdef void create_scheduled(self):
        pass
        
    cdef void create_random(self):
        pass

    cdef np.uint32_t[:] get_nodes_to_type(self,np.uint32_t node, np.uint8_t type):
        cdef Py_ssize_t index 
        cdef np.uint16_t tmp_type
        cdef np.uint8_t tmp_digit 
        cdef np.uint32_t[:] nodes = np.zeros(shape=self.node_np[node, 0, 0]+1, dtype=np.uint32)
        nodes[0] = 0
        for index in range(1, self.node_np[node, 0, 0]):
            tmp_type = self.node_np[node, index, 1]
            while tmp_type >= 1:
                tmp_digit = tmp_type % 10
                if tmp_digit == type + 1:
                    nodes[nodes[0] + 1] = self.node_np[node, index, 0]
                    nodes[0] += 1
                    break
                else:
                    tmp_type = tmp_type // 10
        return nodes
    




