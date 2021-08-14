from data import DataLoader
from network.simulation.network import NetworkSimulator
from simulation import FastSimulation
import configparser
import xmltodict
import pprint
import numpy as np
import gc

class Simulation:

    def __init__(self, cfg_path, is_training, city_size, print_updates):
        network_simulator = NetworkSimulator(city_size, 20, 0.5, "network/notebooks/", verbose=True)
        network_simulator.generate_graph()
        #print(network_simulator.net.edges)
        self.type_dict = {
            'friend': 1,
            'household': 2,
            'colleague': 3,
            'partner': 4,
            'family': 5
        }
        
        nodes = network_simulator.net.nodes
        edges = network_simulator.net.edges
        attributes = np.full(shape=(len(nodes), 3), dtype=np.uint32, fill_value=-1)
        for key, value in nodes.items():
            #print(key, value)
            attributes[int(key), 0] = np.uint32(value['home facility'].replace('"','').split('_')[1])
            if 'edu' in value['work facility']:
                attributes[int(key), 1] = np.uint32(value['work facility'].replace('"','').split('_')[1])
            else:
                attributes[int(key), 2] = np.uint32(value['work facility'].replace('"','').split('_')[1])
        #print(attributes)    
        
        node_counts = np.zeros(shape=city_size, dtype=np.uint32)
        for key, value in edges.items():
            node_counts[int(key[0])] += 1
            node_counts[int(key[1])] += 1
        max_count = np.max(node_counts)
        node_np = np.full(shape=(city_size, max_count + 1, 2), fill_value=-1, dtype=np.uint32)
        node_np[:, 0, :] = 0
        cur_node = 0 
        info_matrix = np.zeros(shape=(city_size, 5), dtype=np.uint8)
        for key, value in edges.items():
            cur_key = int(key[0])
            cur_partner = int(key[1])
            types = value['type']
            type_list = list()
            for type in types:
                info_matrix[cur_key, self.type_dict[type] - 1] = True
                info_matrix[cur_partner, self.type_dict[type] - 1] = True
                type_list.append(self.type_dict[type])
            con_type = self.ctype_to_single_int(type_list)
            node_np[cur_key, node_np[cur_key, 0, 0] + 1, 0] = cur_partner
            node_np[cur_key, node_np[cur_key, 0, 0] + 1, 1] = con_type
            node_np[cur_partner, node_np[cur_partner, 0, 0] + 1, 0] = cur_key
            node_np[cur_partner, node_np[cur_partner, 0, 0] + 1, 1] = con_type
            
            node_np[cur_key, 0, 0] += 1
            node_np[cur_partner, 0, 0] += 1
        print(node_np)
        np.random.seed(1)
        #self.data = DataLoader('../resources/result_graph.xml', 1, 0, '')
        self.config = configparser.ConfigParser()
        self.config.read(cfg_path)
        #legal_dict, population_dict, facilities, infrastructure, personals, simulation_dict, connection_dict
        legal_dict = dict(self.config['LEGAL_CONDITIONS'])
        population_dict = dict(self.config['POPULATION_STATS'])
        facilities = dict(self.config['FACILITIES'])
        infrastructure = dict(self.config['INFRASTRUCTURE'])
        personals = dict(self.config['PERSONAL_FACILITIES'])
        simulation_dict = dict(self.config['SIMULATION'])
        connection_dict = dict(self.config['CONNECTED_ENCOUNTERS'])
        restriction_dict = dict(self.config['RESTRICTIONS'])
        prevention_dict = dict(self.config['PREVENTION'])
        print(node_np.shape[0])
        del network_simulator, nodes, edges
        gc.collect()
        simulation = FastSimulation(is_training, city_size, print_updates, node_np, attributes, info_matrix, legal_dict, population_dict, facilities, infrastructure, personals, simulation_dict, connection_dict)



    def ctype_to_single_int(self, types):
        type_len = len(types)
        val = 0
        for i in range(type_len):
            val += types[i] * pow(10, i)
        return val