import xmltodict
import sys
import numpy as np


# @DavidDrothler
# Loading long-term social graph data from xml file
# Storing xml data in numpy array with restriction to count: lower number for testing purposes, else no specification
# path only used for xml extraction
# Potentially getting dict/array as direct input, bypassing the need to convert data

id_dict = {
    "family": 2,
    "friend": 3,
    "colleague": 4,
    "partner": 5,
    "household": 6
}  # dictionary that we use to assign integer ids to attribute strings.
# any id that is not listed in this dict will lead to errors
# might make this dynamic in the future, if a new connection type is detected, it gets a new id automatically

facility_dict = {
    "home facility": 0,
    "work facility": 1
}
# implementing this in case I need the connection to facilities in the future, not used right now tho


class DataLoader:

    def __init__(self, path, count, mode, export_path):
        self.path = path
        self.count = count
        self.export_path = export_path
        self.mode = mode
        self.data = None
        self.info = None
        print("Getting ", count, " nodes from ", path)
        self.import_from_xml()

    def import_from_xml(self):
        print(self.mode, self.count)
        with open(self.path) as graph:
            self.data = xmltodict.parse(graph.read(), process_namespaces=True)
        original_stdout = sys.stdout
        with open('xml_data_log.txt', 'w') as f:
            sys.stdout = f
            if int(self.mode) == 1:
                print("Testing Purposes")
                data_list = list(self.data["graph"]["nodes"]['node'][0:int(self.count)])
                print(len(data_list))
            else:
                print(self.mode)
                data_list = list(self.data["graph"]["nodes"]['node'])
            self.data = np.array(data_list)
            print("NP array: ", self.data)
            sys.stdout = original_stdout

    def parse_connection_type(self, connection_type):
        types = connection_type.split()
        type_ids = list()
        for string in types:
            if string != ',':   # inconsistent xml input, only some connection properties are separated by commas
                type_ids.append(string)
        return type_ids

    def connection_type_to_int(self, types, dictionary):
        indices = list()
        for type in types:
            indices.append(dictionary[type])

        return indices

    # prepare_data() takes the imported xml file dict and leaves out currently unimportant information
    # each node has various connections, which store the id of the node its connected to, as well as the type of
    # connection. we have N nodes, each node has m_n connections, each connection has d_m_n types
    # our output list is of shape N x M_n x D_m_n and can be represented in 3d space.
    # i might consider creating a uniform 3d matrix with M_n = max(M_n) and D_m_n = max(D_m_n), which would create
    # a lot of zero entries, but could be computationally faster when using numpy

    def prepare_data(self, dictionary):
        converted_nodes = list()
        node_info = list()
        for node in self.data:
            id = int(node['@id'])
            info = list()
            connections = list()
            for attribute in node['attributes']['attribute']:
                info.append(attribute['value'])
            for connection in node['connections']['connection']:
                #print(connection['node_id'], connection['connection_type'])
                connection_id, connection_type = int(connection['node_id']), self.connection_type_to_int(self.parse_connection_type(connection['connection_type']), dictionary)
                connections.append([connection_id, connection_type])
            converted_nodes.append([id, connections])
            node_info.append(info)
        return converted_nodes, node_info

    # pls dont add more than 9 types of connections 1-9 in the Long term social network
    def ctype_to_single_int(self, types):
        type_len = len(types)
        val = 0
        for i in range(type_len):
            val += types[i] * pow(10, i)
        return val

    def prepare_numpy_data(self, nodes, node_info, facility_data):
        node_len = len(nodes)
        max_connections = 0
        # getting maximum connection dimension
        for node in range(node_len):
            connections = nodes[node][1]
            connection_length = len(connections)
            if connection_length > max_connections:
                max_connections = connection_length
        node_np = np.full([node_len, max_connections + 1, 2], fill_value=0, dtype=np.uint32)

        # initializing node numpy data
        for node in range(node_len):
            for connection in range(1, len(nodes[node][1]) + 1):
                # print(nodes[node][1][connection - 1])
                node_np[node][connection - 1][0] = nodes[node][1][connection - 1][0]
                node_np[node][connection - 1][1] = self.ctype_to_single_int(nodes[node][1][connection - 1][1])
                # storing/increasing connection count
                node_np[node][0][0] += 1

        # assuming we only have 3 types of attributes, work, edu and home :)
        facilities = (facility_data['institutions'].replace(' ', '')).split(',')
        info_np = np.full([node_len, len(facilities)], fill_value=-1, dtype=np.uint64)
        #print(node_info)
        for node in range(node_len):
            for index, info in enumerate(node_info[node]):
                #print(info)
                fac_str, id_str = (info.replace('"', '')).split('_')
                id_int = np.uint32(id_str)
                fac_int = facilities.index(fac_str)
                info_np[node][fac_int] = id_int


        return node_np, info_np