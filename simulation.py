import numpy as np
import pandas as pd

from plotting import plots
from producers import P
from consumers import C
from social_charity import SC
from node import Node
    

class Simulation:
    """
    simulation graph

    nodes = list of all nodes ordered by Ps, SCs, Cs
    flows = n x n matrix with np.NaN 
    """
    def __init__(self, horizon: int,
                 Ps: list[P], SCs: list[SC], Cs: list[C]):
        """
        inputs:
            horizon: simulation horizon
            Ps: list of producers
            SCs: list of sharers
            Cs: list of consumers
        """
        self.horizon = horizon
        self.Ps = Ps
        self.SCs = SCs
        self.Cs = Cs
        self.nodes = Ps + SCs + Cs
        self.n_nodes = len(self.nodes)
        self.flows = np.matrix([[np.NaN for i in range(self.n_nodes)] for i in range(self.n_nodes)])  # flows in graph
    
    def update_flows(self, y: np.ndarray, N: Node):
        """
        update flow
        """
        y = y[:-1]  # exclude final row (foodwaste) to get all flows

        N_idx = next(n for n in self.nodes if n.name==N.name)  # index for current node index
        idxs = [n for n in self.nodes if n.name in N.flow_nodes]  # find matching nodes indexes

        if len(idxs)!=y.shape[0]:  # check for errors
            print(idxs)
            print(y)
            raise Exception("idxs shape does not match y shape")
        
        for i in idxs:
            self.flows[N_idx, i] = y[i]   # update flow


    def sim_step(self, pprint=True, plot=True):
        """
        simulate step in graph

        Pseudeocode

        for v in Ps
            y = v.sim_step
            flows <- update flows (y, v)
        for v in SCs:
            y = v.sim_step(flows)
            flows <- update flows (y, v)
        for v in Cs:
            y = v.sim_step(flows)
        """
        out = pd.DataFrame(None, index=[k for k in self.horizon], columns=[n.name for n in self.nodes])

        for k in range(self.horizon):
            for p in self.Ps:  # 1. simulate producers
                y = p.sim_step(k, self.flows)
                self.update_flows(y, p)
                out[p.name, k] = y
            for sc in self.SCs:  # 1. simulate producers
                y = sc.sim_step(k, self.flows)
                self.update_flows(y, sc)
                out[sc.name, k] = y
            for c in self.Cs:
                y = c.sim_step(k, self.flows)
                out[c.name, k] = y

        out.to_csv('out.csv')
