from sklearn.cluster import DBSCAN as dbscan
import numpy as np
import sys


def get_distance_matrix(G):
    """
    G: graph of which we want the distance matrix of the nodes
    return: The distance matrix of all the nodes.
    """
    num_nodes = len(G.nodes)
    result = np.zeros((num_nodes, num_nodes))

    for source_idx, source in enumerate(G.nodes):
        for target_idx, target in enumerate(G.nodes):
            if source_idx == target_idx:
                continue
            try:
                result[source_idx, target_idx] = nx.algorithms.shortest_path(G, source, target)
            except nx.exception.NetworkXNoPath as NP:
                pass
            except Exception as e:
                print("Something went wrong!")
                print(e)
                sys.exit(-1)
    
    return result
            