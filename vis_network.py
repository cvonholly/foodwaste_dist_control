import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from random import random


# load data
df = pd.read_csv('results/flows.csv', index_col=0, header=[0,1])

Gs = []  # create list of graphs

for t in df.columns.get_level_values(0).unique():
    Gs.append(nx.from_pandas_adjacency(df[t].fillna(0)))

G = Gs[-1]

pos = {}

for node in G.nodes():
    x, y = random(), random()
    pos[str(node)] = (x,y)
    G.nodes[node]['pos'] = {x, y}  # create random coordinates

print(G.edges(data=True))

options = {
    'node_color': 'blue',
    'node_size': 10,
    'width': 3,
    'arrowstyle': '-|>',
    'arrowsize': 12,
    'with_labels': True
}

nx.draw_networkx(G, arrows=True, **options)

edge_labels = dict([((u,v,), f"{d['weight']:.2f}") for u,v,d in G.edges(data=True)]) #nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)

plt.show() 



exit()
# subax1 = plt.subplot(121)
# nx.draw(G)   # default spring_layout
# subax2 = plt.subplot(122)
# nx.draw(G, pos=nx.circular_layout(G), node_color='r', edge_color='b')

#
# copied from:  https://plotly.com/python/network-graphs/
#

# exit()

# G = nx.random_geometric_graph(200, 0.125)


node_text = []
for node in G.nodes():
    G.nodes[node]['pos'] = {random(), random()}  # create random coordinates
    G.nodes[node]['name'] = str(node)
    node_text.append(str(node))

print(list(G.nodes(data=True)))

# exit()

# print(G.nodes[0])
# print(G.nodes['P1'])

edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='YlGnBu',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))


# color node points
# node_adjacencies = []
# for node, adjacencies in enumerate(G.adjacency()):
#     node_adjacencies.append(len(adjacencies[1]))
    # node_text.append('# of connections: '+str(len(adjacencies[1])))
# for node in G.nodes():
#     node_text.append(node['name'])

# node_trace.marker.color = node_adjacencies


node_trace.text = node_text

print("create network graph")

# create network graph
fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.show()
