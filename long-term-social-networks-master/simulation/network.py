import networkx as nx
import numpy as np

class Network:

    def __init__(self, init_net=None):
        self.graph = init_net if init_net else nx.DiGraph()
        self.K = 2
        self.p_threshold = 0.5
        self.f1 = 0
        self.f2 = 0

    def add_node(self, id, m):
        # TODO: how to decide to which m nodes we connect?
        self.graph.add_node(id)
        in_degree = np.array(self.graph.in_degree)[:,1]
        t0, t1, t2 = self.determine_regression_factors()
        a = self.close_degree()
        for g in self.graph.nodes:
            if id != g and self.connection_proba(id, g, in_degree, a, t0, t1, t2) >= self.p_threshold:
                self.graph.add_edge(id, g)


    def add_connections(self):
        in_degree = np.array(self.graph.in_degree)[:,1]
        t0, t1, t2 = self.determine_regression_factors()
        a = self.close_degree()
        con_pr = self.connection_proba_matrix(in_degree, a, t0, t1, t2) >= self.p_threshold
        for (i, g) in np.argwhere(con_pr):
            if i!=g:
                self.graph.add_edge(i, g)

    def delete_connections(self):
        in_degree = np.array(self.graph.in_degree)[:,1]
        t0, t1, t2 = self.determine_regression_factors()
        a = self.close_degree()
        # only iterate over existing edges to reduce complexity
        for (i, g) in self.graph.edges:
            pr = self.connection_proba(i, g, in_degree, a, t0, t1, t2)
            print(pr)
            if pr < self.p_threshold:
                self.graph.remove_edge(i, g)

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


    def close_degree(self):
        # TODO
        return np.ones((4,4), dtype=float)

    def determine_regression_factors(self):
        # TODO
        t0 = 0.2
        t1 = 0.2
        t2 = 0.2
        return t0, t1, t2

