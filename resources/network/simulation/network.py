import networkx as nx
import numpy as np
import yaml
import random
from collections import Counter
import sys

sys.stdout = open("output.txt", "w")

class Network:

    def __init__(self, init_net=None, p_bernoulli=0.5):
        self.graph = init_net if init_net else nx.DiGraph()
        self.K = 2
        self.p_threshold = 0.5
        self.f1 = 0
        self.f2 = 0
        self.p_bernoulli = p_bernoulli
        self.node_ids = [] # I suggest we keep this for performance issues, O(N^2) is not acceptable for a graph of size 1.000.000
                            # we use the list to draw a random number of nodes to connect to on O(1)

    def add_node(self, node_id, **node_data):
        """
        node_id: The id with which the node is identified
        node_data: A dictionary containing data about the node
        """

        # TODO: how to decide to which m nodes we connect?
        self.graph.add_node(node_id, **node_data) #directly add all the properties for this node

        

        for g in self.graph.nodes:
            if node_id != g and np.random.binomial(size=1, n=1, p=self.p_bernoulli)[0] == 1:
                self.graph.add_edge(node_id, g)
            if node_id != g and np.random.binomial(size=1, n=1, p=self.p_bernoulli)[0] == 1:
                self.graph.add_edge(g, node_id)
        return self.graph


    def add_connections(self):
        nodes = list(self.graph.nodes)

        for node_id1 in nodes:
            for node_id2 in nodes:
                if node_id1 != node_id2:
                    if np.random.binomial(size=1, n=1, p=self.p_bernoulli)[0] == 1:
                        self.graph.add_edge(node_id1, node_id2)


    def delete_connections(self):
        nodes = list(self.graph.nodes)

        for node_id1 in nodes:
            for node_id2 in nodes:
                if node_id1 != node_id2:
                    if np.random.binomial(size=1, n=1, p=self.p_bernoulli)[0] == 1:
                        try:
                            self.graph.remove_edge(node_id1, node_id2)
                        except nx.exception.NetworkXException as ne:
                            pass
    

    def connection_proba(self, i, g, k, a, theta0, theta1, theta2):
        """
        Calculate the connection probability for two nodes i and g based on the paper's fomula
        :param i: node from
        :param j: node to
        :param K: threshold for node's in-degree
        :param k: node i's in-degree
        :param a: close degree between nodes i and g
        :param theta0, theta1, theta2: regression factors
        :return: probability
        """
        proba = theta0
        proba += theta1 * a[i][g] * (1- (k[i]/self.K))
        proba += theta2 * k[i]
        proba = np.exp(-proba)
        proba = 1 / (1 + proba)
        return proba

    def connection_proba_matrix(self, k, a, t0, t1, t2):
        a_scaled = t1 * a
        k_scaled = 1 - (k / self.K)
        pr = t0 + a_scaled * k_scaled[:, None]
        k_scaled = t2 * k
        pr += k_scaled[:, None]
        pr = np.exp(-pr)
        pr = 1 / (1 + pr)
        return pr


    def close_degree(self, node1_id, node2_id):
        

        return np.ones((4,4), dtype=float)
    
    def _aij_t(self, node1_id, node2_id):
        pass

    def determine_regression_factors(self):
        # TODO
        t0 = 0.2
        t1 = 0.2
        t2 = 0.2
        return t0, t1, t2



"""
This class is the Social Network simulator used to generate a Watts Strogatz model
based on data supplied about the population
"""
class NetworkSimulator:

    def __init__(self, num_nodes, k_degree, p_probability, population_data_path, data_duplicates=False, verbose=False):

        self.num_nodes = num_nodes
        self.k_degree = k_degree
        self.p_probability = p_probability
        self.population_data_path = population_data_path
        self.data_duplicates = data_duplicates
        self.net = None
        self.verbose = verbose

        with open('config.yaml', 'r') as stream:
            self.config = config = yaml.safe_load(stream)
    
    def _get_random_graph(self):
        # no final model, use random graph
        #net = nx.read_gexf('graph.gexf')
        try:
            if self.verbose:
                print("Creating the graph...", flush=True)
            self.net = nx.watts_strogatz_graph(self.num_nodes, self.k_degree, self.p_probability)
            self.net = nx.relabel_nodes(self.net, lambda x: str(x))
        except Exception as e:
            print(f"Random graph generation failed: \n{e}\n", flush=True)
            sys.exit(-1)
        return self.net

    def load_facility_files(self):
        try:
            if self.verbose:
                print("Loading the facilities...", flush=True)

            if self.data_duplicates:
                work = list(np.genfromtxt(self.population_data_path + 'work_facilities_withduplicates.txt', dtype='str'))
                edu = list(np.genfromtxt(self.population_data_path + 'education_facilities_withduplicates.txt', dtype='str'))
            else:
                work = list(np.genfromtxt(self.config['work_data'], dtype='str'))
                edu = list(np.genfromtxt(self.config['education_data'], dtype='str'))
            homes = list(np.genfromtxt(self.config['home_data'], dtype='str'))
        except Exception as e:
            print(f"Loading facilities failed: \n{e}\n")
            sys.exit(-1)
        return work + edu, homes

    def propagate_labels(self, attribute_name, attribute_values):
        # init some nodes with labels
        try:
            if self.verbose:
                print("Propagating the labels...", flush=True)
            max_id = len(self.net.nodes()) - 1
            labels_one_hot = {}
            for i, attr in enumerate(attribute_values):
                node_id = str(random.randint(0, max_id))
                self.net.nodes[node_id][attribute_name] = attr
                #labels_one_hot[node_id] = i
            # propagate label through graph
            print('Propagate labels, this can take a while')
            labels = list(nx.algorithms.node_classification.local_and_global_consistency(self.net, label_name=attribute_name))
            #labels = list(custom_label_propagation(net, attribute_name, np.array(list(labels_one_hot.values()))))
            #labels = efficient_node_classification(net, label_name=attribute_name)
            graph_labels_counter = dict(Counter(labels))
            for node in self.net:
                self.net.nodes[node][attribute_name] = labels[int(node)]
            if self.verbose:
                print("Labels propagated!", flush=True)
        except Exception as e:
            print(f"Label propagation failed: \n{e}\n")
            sys.exit(-1)

        return graph_labels_counter
    
    def generate_graph(self):
        if self.verbose:
            print("Generating graph...", flush=True)
        self._get_random_graph()
        self.load_facility_files()

        self._get_random_graph()
        works, homes = self.load_facility_files()

        work_counter = self.propagate_labels('work facility', works)
        house_counter = self.propagate_labels('home facility', homes)

        
        try:
            for node, neighbor in self.net.edges:
                self.net[node][neighbor]['type'] = set()

                #flag to check if a label was assigned
                flag = False
                # add colleague label
                if self.net.nodes[node]['work facility'] == self.net.nodes[neighbor]['work facility']:
                    self.net[node][neighbor]['type'].add('colleague')
                    flag = True

                # add household label
                elif self.net.nodes[node]['home facility'] == self.net.nodes[neighbor]['home facility']:
                    if random.random() <= self.config['q_houesehold']:
                        self.net[node][neighbor]['type'].add('household')
                        flag = True

                # add family label
                if random.random() <= self.config['q_family']:
                    self.net[node][neighbor]['type'].add('family')
                    continue
                # add family label
                if random.random() <= self.config['q_partner']:
                    self.net[node][neighbor]['type'].add('partner')
                    continue
                
                if random.random() <= self.config['q_friend']:
                    self.net[node][neighbor]['type'].add('friend')
                    flag = True

                #If no other relationship was added make him/her a friend
                if flag == False:
                    self.net[node][neighbor]['type'].add('friend')
        except Exception as e:
            print(f"Multi-edge stage failed: \n{e}\n")
        

if __name__ == "__main__":
    # Create the object, note the config.yaml file for the configuration
    import sys
    sys.stdout = open("output.txt", "w")

    network_simulator = NetworkSimulator(118820, 20, 0.5, "../notebooks/", verbose=True)
    #generate the graph
    network_simulator.generate_graph()

    
    nx.write_gpickle(network_simulator.net, "graph.gexf")
    print(len(network_simulator.net.nodes()))