import networkx as nx
import yaml
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
from utils import *
from scipy.sparse import csr_matrix


def generate_random_graph(num_nodes, k, p):
    # no final model, use random graph
    #net = nx.read_gexf('graph.gexf')
    net = nx.watts_strogatz_graph(num_nodes, k, p)
    net = nx.relabel_nodes(net, lambda x: str(x))
    return net

def load_facility_files(population_data, duplicates=False):
    if duplicates:
        work = list(np.genfromtxt(population_data + 'work_facilities_withduplicates.txt', dtype='str'))
        edu = list(np.genfromtxt(population_data + 'education_facilities_withduplicates.txt', dtype='str'))
    else:
        work = list(np.genfromtxt(config['work_data'], dtype='str'))
        edu = list(np.genfromtxt(config['education_data'], dtype='str'))
    homes = list(np.genfromtxt(config['home_data'], dtype='str'))
    return work + edu, homes

def propagate_labels(attribute_name, attribute_values):
    # init some nodes with labels
    max_id = len(net.nodes()) - 1
    label_dict = {}
    for i, attr in enumerate(attribute_values):
        node_id = str(random.randint(0, max_id))
        net.nodes[node_id][attribute_name] = attr
        label_dict[node_id] = i
    # propagate label through graph
    #labels = list(nx.algorithms.node_classification.local_and_global_consistency(net, label_name=attribute_name))
    #labels = custom_label_propagation(net, label_dict)
    labels = efficient_node_classification(net, label_name=attribute_name)
    for node in net:
        net.nodes[node][attribute_name] = labels[int(node)]

def plot_node_example():
    labels = nx.get_node_attributes(net.subgraph(['0'] + list(net['0'])), 'work facility')
    pos = nx.spring_layout(net.subgraph(['0'] + list(net['0'])), scale=2)
    nx.draw(net.subgraph(['0'] + list(net['0'])), labels=labels, pos=pos)
    grafo_labels = nx.get_edge_attributes(net.subgraph(['0'] + list(net['0'])), 'type')
    nx.draw_networkx_edge_labels(net.subgraph(['0'] + list(net['0'])), pos, edge_labels=grafo_labels)
    plt.show()

# ---------------------------------------
# 0. Load config file with parameters
with open('config.yaml', 'r') as stream:
    config = config = yaml.safe_load(stream)

# 1. Load network model
net = generate_random_graph(config['num_nodes'], config['avg_degree'], config['p_reconnect'])
print('Generated network')

# 2. load facilties
works, homes = load_facility_files(config['population_data_path'])
print('Loaded facility values')

# 3. Propagate work/education facilities
print('Propagating node attributes, this will take some time')
propagate_labels('work facility', works)
print('Propagated work')

# 4. Propagate house facilities
propagate_labels('home facility', homes)
print('Propagated node attributes')


# 4.1 compare facility distribution with original
#facilities_dup = load_facility_files(duplicates=True)
#orig_labels_counter = dict(Counter(facilities_dup))
#for facility in work_counter:
#    if abs(work_counter[facility] - orig_labels_counter[facility]) > 5:
#        print(facility, work_counter[facility], orig_labels_counter[facility])

# 5. Label edges
for node, neighbor in net.edges:
    net[node][neighbor]['type'] = set()

    #flag to check if a label was assigned
    flag = False
    # add colleague label
    if net.nodes[node]['work facility'] == net.nodes[neighbor]['work facility']:
        net[node][neighbor]['type'].add('colleague')
        flag = True

    # add household label
    elif net.nodes[node]['home facility'] == net.nodes[neighbor]['home facility']:
        if random.random() <= config['q_houesehold']:
            net[node][neighbor]['type'].add('household')
            flag = True

    # add family label
    if random.random() <= config['q_family']:
        net[node][neighbor]['type'].add('family')
        continue
    # add family label
    if random.random() <= config['q_partner']:
        net[node][neighbor]['type'].add('partner')
        continue
    
    if random.random() <= config['q_friend']:
        net[node][neighbor]['type'].add('friend')
        flag = True

    #If no other relationship was added make him/her a friend
    if flag == False:
        net[node][neighbor]['type'].add('friend')


print('Labeled edges')

# 6. Plot a node example
#plot_node_example()

# 5. Export graph file
nx.write_gpickle(net, 'multi_edge_graph_1pct.gexf')
print('Finished')