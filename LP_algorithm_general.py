import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from params.params import load_params

if __name__=="__main__":
    # load parameters
    name = "SCS"
    params = load_params(name)
    n_ps = params["n_ps"] # number of producers
    n_cs = params["n_cs"] # number of consumers
    n_scs = params["n_scs"] # number of social-charities
    T = params["T"] # numbers of time steps / states
    horizon = params["horizon"] # numbers of simulation steps
    T_start_sc = 2 # defines state where social charities start to receive food from producers
    source_output = np.sum(params["input_flows"]) # p["input_flows"].sum() (=demand of source node)
    daily_food_intake = params["food_intake"] # p["food_intake"] (=capacity of edge from consumption node to sink!)
    x0_p = params["x0_p"] # initial state of producer
    x0_c = params["x0_c"] # initial state of consumer
    x0_sc = params["x0"] # initial state of social charitie (do we actually need this???)


# n_ps = 1 # number of producers
# n_cs = 1 # number of consumers
# n_scs = 1 # number of social-charities
# T = 5 # numbers of time steps / states
# horizon = 3 # numbers of simulation steps
# T_start_sc = 2 # defines state where social charities start to receive food from producers
# source_output = 6 # p["input_flows"].sum() (=demand of source node)
# daily_food_intake = 3 # p["food_intake"] (=capacity of edge from consumption node to sink!)
# x0_p = np.ones((T, 1)) # initial state of producer
# x0_c = np.ones((T, 1)) # initial state of consumer
# x0_sc = np.zeros((T, 1)) # initial state of social charitie (do we actually need this???)



"""
parameters needed for this network:


params name: "SCS"

p["input_flows"].sum()  # source output
T  # numbers of time steps / states
n_ps  # number of producers
n_cs  # number of consumers
n_scs  # number of social-charities
p["food_intake"]   # daily food intake required by consumers: implement in demand

notes:
- capacities: set high enough to allow for all possible flows
- weights: set 0 for non-foodwaste, set 1 for foodwaste
- demand: set food_intake for consumers for every time step
"""

# CREATE DIRECTED GRAPH
G = nx.DiGraph()

# CREATE NODES OF DIRECTED GRAPH

# Create source and sink node of graph to preserve mass conservation
G.add_node('Source',pos=(-2,-10),demand = -(source_output))
#G.add_node('Sink',pos=(12,-10),demand = source_output + x0_p.sum() + x0_c.sum() + x0_sc.sum())

# Create node representing food waste
G.add_node('Food waste',pos=(3,-12), demand = (source_output + x0_p.sum() + x0_c.sum() + x0_sc.sum()) - (n_cs*daily_food_intake*horizon))

# Create nodes for producer states
for n in range(1, n_ps + 1):
    for m in range (T):
        G.add_node(f"Producer_{n}_state_{m}",pos=(0,-(6*n + m)))

# Create nodes for consumer states
for n in range(1, n_cs + 1):
    for m in range (T):
        G.add_node(f"Consumer_{n}_state_{m}",pos=(6,-(6*n + m)))

# Create nodes for social charity states
for n in range(1, n_scs + 1):
    for m in range (T_start_sc, T): # range defines from what state sc start to receive food products
        G.add_node(f"Socialcharity_{n}_state_{m}",pos=(3,-(6*n + (m+0.5))))

# Create nodes representing consumption of each consumer per day
for n in range(1, n_cs + 1):
    for m in range (1, horizon+1):
        G.add_node(f"Consumption_{n}_day_{m}",pos=(9,-(6*n + m)),demand = daily_food_intake)

# CREATE EDGES OF DIRECTED GRAPH

# Add edges connecting source node to all consumer state 0 nodes
for n in range(1, n_ps + 1):
    G.add_edge('Source',f"Producer_{n}_state_{0}", weight = 0, capacity=10000)

# Add edges connecting daily consumption nodes to sink
#for n in range(1, n_cs + 1):
#    for m in range (1, horizon+1):
#        G.add_edge(f"Consumption_{n}_day_{m}", 'Sink', weight = 0, capacity=daily_food_intake)

# Add edges representing food that stays by producers (e.g. transition from state m to state m+1)
for n in range(1, n_ps + 1):
    for m in range (T-1):
        G.add_edge(f"Producer_{n}_state_{m}",f"Producer_{n}_state_{m+1}", weight = 0, capacity=10000)

# Add edges representing food that stays by consumers (e.g. transition from state m to state m+1)
for n in range(1, n_cs + 1):
    for m in range (T-1):
        G.add_edge(f"Consumer_{n}_state_{m}",f"Consumer_{n}_state_{m+1}", weight = 0, capacity=10000)

# Add edges representing food that stays by social charitys (e.g. transition from state m to state m+1)
for n in range(1, n_scs + 1):
    for m in range (T_start_sc, (T-1)):
        G.add_edge(f"Socialcharity_{n}_state_{m}",f"Socialcharity_{n}_state_{m+1}", weight = 0, capacity=10000)

# Add edges representing food flow from producers to consumers
for n in range(1, n_ps + 1):
    for k in range(1, n_cs + 1):
        for m in range (T-1):
            G.add_edge(f"Producer_{n}_state_{m}",f"Consumer_{k}_state_{m}", weight = 0, capacity=1)

# Add edges representing food flow from producers to socialcharities
for n in range(1, n_ps + 1):
    for k in range(1, n_scs + 1):
        for m in range (T_start_sc, (T-1)):
            G.add_edge(f"Producer_{n}_state_{m}",f"Socialcharity_{k}_state_{m}", weight = 0, capacity=1)

# Add edges representing food flow from socialcharities to consumers
for n in range(1, n_scs + 1):
    for k in range(1, n_cs + 1):
        for m in range (T_start_sc, (T-1)):
            G.add_edge(f"Socialcharity_{n}_state_{m}",f"Consumer_{k}_state_{m}", weight = 0, capacity=10000)

# Add edges representing daily consumption of consumers
for n in range(1, n_cs + 1):
    for k in range(1, horizon + 1):
        for m in range (T-1):
            G.add_edge(f"Consumer_{n}_state_{m}",f"Consumption_{n}_day_{k}", weight = 0, capacity=10000)

# Add edges representing food waste from producers
for n in range(1, n_ps + 1):
    G.add_edge(f"Producer_{n}_state_{T-1}",f"Food waste", weight = 1, capacity=10000)

# Add edges representing food waste from cosnumers
for n in range(1, n_cs + 1):
    G.add_edge(f"Consumer_{n}_state_{T-1}",f"Food waste", weight = 1, capacity=10000)

# Add edges representing food waste from producers
for n in range(1, n_scs + 1):
    G.add_edge(f"Socialcharity_{n}_state_{T-1}",f"Food waste", weight = 1, capacity=10000)

# Draw edge connecting food waste node with sink
#G.add_edge("Food waste","Sink", weight = 0, capacity=10000)


# State initialization of producer nodes:
for n in range(1, n_ps + 1):
    for m in range (T):
        G.add_node(f"Initial_Producer_{n}_state_{m}",pos=(-1,-(6*n + m)+0.5), demand = -(x0_p[m][0]))
        G.add_edge(f"Initial_Producer_{n}_state_{m}",f"Producer_{n}_state_{m}", weight = 0, capacity=10000)

# State initialization of consumer nodes:
for n in range(1, n_cs + 1):
    for m in range (T):
        G.add_node(f"Initial_Consumer_{n}_state_{m}",pos=(5,-(6*n + m)+0.5), demand = -(x0_c[m][0]))
        G.add_edge(f"Initial_Consumer_{n}_state_{m}",f"Consumer_{n}_state_{m}", weight = 0, capacity=10000)


# DRAW GRAPH REPRESENTATION OF LP PROBLEM

# Draw nodes of graph according to given position coordinates
pos = nx.get_node_attributes(G,'pos') # positions for all nodes
labels = {node: node for node in G.nodes()}  # labels for all nodes

nx.draw(G, pos, with_labels=True, labels=labels, node_size=1000, node_color='lightblue', font_size=7, font_weight='bold')

# Draw edges with capacities and costs
edge_labels = {(u, v): f"{d['capacity']}, {d['weight']}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=6)

plt.show()


# COMPUTE MINIMUM-COST-FLOW OF NETWORK

flowCost, flowDict = nx.network_simplex(G)

print('Minimum cost:', flowCost)
print(' ')

for key_i, inner_dict in flowDict.items():
    for key_j, inner_val in inner_dict.items():
        print(f'{key_i}->{key_j} \t Flow: {inner_val}')

