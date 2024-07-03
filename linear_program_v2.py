import networkx as nx
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus, value
import matplotlib.pyplot as plt
from LP_algorithm import min_cost_flow



from params.params import load_params


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


if __name__=="__main__":
    # load parameters
    name = "SCS"
    params = load_params(name)

# for n in n_ps:



# Create directed graph
G = nx.DiGraph()

# Add nodes
G.add_node('Producer1')
G.add_node('Producer2')
G.add_node('Producer3')
G.add_node('Producer4')
G.add_node('Consumer1')
G.add_node('Consumer2')
G.add_node('Consumer3')
G.add_node('Consumer4')
G.add_node('Charity2')
G.add_node('Charity3')
G.add_node('Charity4')
G.add_node('Food_waste')
G.add_node('Consumption')
G.add_node('Source', demand = -12)
G.add_node('Sink', demand = 12)


# Add edges with capacities and costs
G.add_edge('Source', 'Producer1', weight = 0, capacity=100)

G.add_edge('Producer1', 'Consumer1', weight = 1, capacity=5)
G.add_edge('Producer1', 'Producer2', weight = 10, capacity=100)
G.add_edge('Producer2', 'Consumer2', weight = 1, capacity=5)
G.add_edge('Producer2', 'Charity2', weight = 0.5, capacity=1)
G.add_edge('Producer2', 'Producer3', weight = 10, capacity=100)
G.add_edge('Producer3', 'Consumer3', weight = 1, capacity=5)
G.add_edge('Producer3', 'Charity3', weight = 6, capacity=3)
G.add_edge('Producer3', 'Producer4', weight = 10, capacity=100)
G.add_edge('Producer4', 'Food_waste', weight = 100, capacity=100)

G.add_edge('Charity2', 'Consumer2', weight = 0.3, capacity=100)
G.add_edge('Charity2', 'Charity3', weight = 10, capacity=100)
G.add_edge('Charity3', 'Consumer3', weight = 3, capacity=100)
G.add_edge('Charity3', 'Charity4', weight = 10, capacity=100)
G.add_edge('Charity4', 'Food_waste', weight = 100, capacity=100)

G.add_edge('Consumer1', 'Consumption', weight = 0, capacity=4)
G.add_edge('Consumer1', 'Consumer2', weight = 10, capacity=100)
G.add_edge('Consumer2', 'Consumption', weight = 0, capacity=4)
G.add_edge('Consumer2', 'Consumer3', weight = 10, capacity=100)
G.add_edge('Consumer3', 'Consumption', weight = 0, capacity=4)
G.add_edge('Consumer3', 'Consumer4', weight = 10, capacity=100)
G.add_edge('Consumer4', 'Food_waste', weight = 100, capacity=100)

G.add_edge('Consumption', 'Sink', weight = 0, capacity=10)
G.add_edge('Food_waste', 'Sink', weight = 0, capacity=100)


# Manually specify node positions
pos = {
    'Producer1': (0, 6),  # position for Producer state 1
    'Producer2': (0, 4),  # position for Producer state 2
    'Producer3': (0, 2),   # position for Producer state 3
    'Producer4': (0, 0),   # position for Producer state 4

    'Charity2': (3, 3),  # position for Charity state 2
    'Charity3': (3, 1),   # position for Charity state 3
    'Charity4': (3, -1),   # position for Charity state 4
    

    'Consumer1': (6, 6),  # position for Consumer state 1
    'Consumer2': (6, 4),  # position for Consumer state 2
    'Consumer3': (6, 2),   # position for Consumer state 3
    'Consumer4': (6, 0),   # position for Consumer state 4
    
    'Consumption': (10, 5),  # position for Consumption node
    'Food_waste': (3, -3),  # position for Food waste node

    'Source': (-3, 3),
    'Sink': (12, 3)
}

# Draw graph
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)

# Draw edges with capacities and costs
edge_labels = {(u, v): f"{d['capacity']}, {d['weight']}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=6)

# Display the graph
plt.show()

flowCost, flowDict = nx.network_simplex(G)

print('Minimum cost:', flowCost)
print(' ')

for key_i, inner_dict in flowDict.items():
    for key_j, inner_val in inner_dict.items():
        print(f'{key_i}->{key_j} \t Flow: {inner_val}')
