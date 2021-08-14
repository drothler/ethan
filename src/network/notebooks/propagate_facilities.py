import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter
from tqdm import tqdm


def load_graph_model():
    # no final model, use random graph
    net = nx.read_gexf('graph.gexf')
    return net

def load_facility_files(duplicates=False):
    if duplicates:
        work = list(np.genfromtxt('population_data/work_facilities_withduplicates.txt', dtype='str'))
        edu = list(np.genfromtxt('population_data/education_facilities_withduplicates.txt', dtype='str'))
    else:
        work = list(np.genfromtxt('population_data/work_facilities_withoutduplicates.txt', dtype='str'))
        edu = list(np.genfromtxt('population_data/education_facilities_withoutduplicates.txt', dtype='str'))
    return work + edu


# ---------------------------------------
# 1. Load network model
net = load_graph_model()
print('Loaded network successfully')

# 2. load facilties
facilities = load_facility_files()
print(facilities)

# 3. Init labels
max_id = len(net.nodes()) -1
for facility in tqdm(facilities):
    node_id = str(random.randint(0, max_id))
    net.nodes[node_id]['facility'] = facility

# 4. Propagate labels
labels = list(nx.algorithms.node_classification.local_and_global_consistency(net, label_name='facility'))
graph_labels_counter = dict(Counter(labels))
for node in tqdm(net):
    net.nodes[node]['facility'] = labels[int(node)]

# 4.1 compare facility distribution with original
facilities_dup = load_facility_files(duplicates=True)
orig_labels_counter = dict(Counter(facilities_dup))
for facility in tqdm(graph_labels_counter):
    if abs(graph_labels_counter[facility] - orig_labels_counter[facility]) > 5:
        print(facility, graph_labels_counter[facility], orig_labels_counter[facility])

# 5. Export graph file
nx.write_gexf(net, 'graph_with_facilities.gexf')
print('Finished')