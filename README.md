# distributed network of foodwaste relevant actors

this project is made in context of the lecture [Advanced Topics in Control](https://www.vvz.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?semkez=2024S&ansicht=ALLE&lerneinheitId=178230&lang=de)

Our idea is to model the system as a distributed network with inflows and outflows (both representing food flow). Each subsystem in the network represents an actor in the food waste cycle. For example, a consumer subsystem consists of a state x representing stored/bought quantities of food and an output of consumption and food waste. 

pre:
- Python
- packages: pandas, dash, plotly

run:
- `main.py` for running simulation
- `vis_results.py` for visualizing results
- `vis_network.py` for visualizing network

## To dev:
- fix marks matrices
- serialize adding new nodes
- normalize certain matrices
    - P flows matrix

- flows have to be adapted

license: MIT

author: Carl von Holly-Ponientzietz
