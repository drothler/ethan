from data import DataLoader

from scheduling import FacilityManager, HumanManager
from simulation import FastSimulation
import disease
import configparser
import json
import pickle
import numpy as np
import array
from pympler import asizeof

# Global attributes that are read from the config file
mode = 1  # 0 if full simulation, 1 for testing purposes
test_size = 100  # sample size, only important when mode set to 1
input_format = ""  # if we ever change the xml output format of the LSTM network
rest_day = 0    # which day of the week are workers least likely to work, usually sunday but might differ
rest_day_probability = 0.8  # probability of a single worker not working on rest days
work_hours_mean = 8     # how many hours an average worker works per day
sleep_hours_default = 7     # how many hours an average person sleeps per day
config = None  # config file

danger_levels = dict()
encounters = list()




def get_config_file(path):
    global config, mode, test_size, input_format, rest_day, rest_day_probability, work_hours_mean, sleep_hours_default
    config = configparser.ConfigParser()
    config.read(path)
    mode = config['SIMULATION']['mode']
    test_size = config['SIMULATION']['test_size']
    rest_day = config['SIMULATION']['rest_day']
    rest_day_probability = config['SIMULATION']['rest_day_probability']
    work_hours_mean = config['SIMULATION']['work_hours_mean']
    sleep_hours_default = config['SIMULATION']['sleep_hours_default']

def xml_to_encounters(file):
    with open(file, 'r') as enc_file:
        print('Found file')
        for index, line in enumerate(enc_file):
            content = '{' + '\"encounters\": ['+ line.replace('][', '], [') + ']}'
            content = json.loads(content)
            for index, encounter in enumerate(content['encounters']):
                content['encounters'][index][0] = int(content['encounters'][index][0])
            encounters.append(content['encounters'])



get_config_file("../resources/config.cfg")

facility_dict = dict(config['FACILITIES'])
infrastructure_dict = dict(config['INFRASTRUCTURE'])
personal_dict = dict(config['PERSONAL_FACILITIES'])
legal_dict = dict(config['LEGAL_CONDITIONS'])
population_dict = dict(config['POPULATION_STATS'])
connection_dict = dict(config['CONNECTED_ENCOUNTERS'])
simulation_dict = dict(config['SIMULATION'])

# TODO: pls automate before pushing
type_dict = {
    'friend': 1,
    'household': 2,
    'colleague': 3,
    'partner': 4,
    'family': 5
}

simulation = FastSimulation(True, 100000, True)

data = DataLoader(path="../resources/result_graph.xml", count=test_size, mode=mode, export_path="")
nodes, info = data.prepare_data(type_dict)
node_np, info_np = data.prepare_numpy_data(nodes, info, personal_dict)
print(np.array(range(0, len(nodes))))

#print(facility_dict)
np.random.seed(0)
#manager = FacilityManager(facility_dict, infrastructure_dict, personal_dict, 118000)
#manager.return_facilities_by_district()
# manager.print_schedules()

person_manager = HumanManager(np.array(range(0, len(nodes)), dtype=np.uint32), node_np, info_np, legal_dict, population_dict, facility_dict, infrastructure_dict, personal_dict, simulation_dict, int((int(infrastructure_dict['districts_per_million']) / 1e6) * len(nodes)))
person_manager.initialize_node_info(np.full(fill_value=0.01, dtype=np.float32, shape=100))
#graph_data, node_info = data.prepare_data()
#infections = dict()
#quarantine = dict()
#isolation = dict()
#immune = set()
#cured = set()

#for node in graph_data:
    #infections[node[0]] = [True, 0]
    #quarantine[node[0]] = [False, 0]
    #isolation[node[0]] = [False, 0]

# node 0 is patient zero
#infections[0] = [True, 0]

# some debugging and testing infections numbers
#ground_truth = [1, 2, 2, 5, 9, 14]



#print(facility_test.get_size())
for i in range(0, 24):
    for j in range(0, 7):
        for x in range(0, 100):
            pass
            # facility_test.add_node_at_time(day = j, hour=i, id=x)




#disease_spread = disease.DiseaseSpread(infections=infections, quarantine=quarantine, isolation=isolation, immune=immune, cured=cured)




#simulate_for_days(7)
#with open('outfile', 'rb') as fp:
    #current_encounters = pickle.load(fp)
    #print('Un-pickling successful')
    #print(current_encounters[0])
#disease_spread.update_encounters(current_encounters)
#print(disease_spread.step())
#disease_spread.update_node_status()
#print(disease_spread.get_total_infections())
#disease_spread.backward(50, len(graph_data), disease_spread.return_maxdim())
