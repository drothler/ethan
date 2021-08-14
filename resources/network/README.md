# Group of Long-term Social Networks
## Most recent updates about our work will be poster here

## Members: Aldi Topalli, Miriam Anschütz (alphabetically ordered)

## How to use the simulation code

We have developed a *NetworkSimulator* class which makes all the neccessary calls to create the network. To use it simply do the following:
```
from simulation.network import NetworkSimulator

network_simulator = NetworkSimulator(100, 20, 0.5, "../notebooks/")
graph = network_simulator.generate_graph()
```
Then the graph can be saved as a pickle by using the following method:
```
import networkx as nx
nx.write_gpickle(graph, "pickle.gexf")
```
 ## Config

 To manipulate the graph properties, besides the parameters passed to the constructer, you can use the ```config.yaml``` file.
 ```
num_nodes: 118820
avg_degree: 20
p_reconnect: 0.5
population_data_path: '../notebooks/population_data/'
#education_data: 'population_data/unique_education_facilities_0_1.txt'
education_data: '../notebooks/population_data/unique_education_facilities1pct.txt'
#work_data: 'population_data/unique_work_facilities_0_1.txt'
work_data: '../notebooks/population_data/unique_work_facilities_1pct.txt'
#home_data: 'population_data/unique_home_facilities_0_1.txt'
home_data: '../notebooks/population_data/unique_home_facilities1pct.txt'
q_houesehold: 0.5
q_family: 0.35
q_partner: 0.05
q_friend: 0.15

 ```

 This is where the simulation method takes the parameters for edge types and the files for the population information.