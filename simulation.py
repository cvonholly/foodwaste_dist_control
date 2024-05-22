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
        self.names = [n.name for n in self.nodes]
        self.n_nodes = len(self.nodes)
        self.flows = pd.DataFrame(np.NAN, index=self.names, columns=self.names)  # flows in graph
        self.out = pd.DataFrame()
    
    def update_flows(self, y: np.ndarray, N: Node):
        """
        update flow
        """
        y = y[:-1]  # exclude final row (foodwaste) to get all flows

        if len(N.flow_nodes)!=y.shape[0]:  # check for errors
            print(N.flow_nodes)
            print(y)
            raise Exception("idxs shape does not match y shape")
        
        # for i in range(len(N.flow_nodes)):
        print("updating flows for ", N.name)
        for i in range(len(N.flow_nodes)):
            self.flows.loc[N.name, N.flow_nodes[i]] = y[i]   # update flow


    def simulate(self, store=True):
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
        c1, c2 = [], []
        for n in self.nodes:
            c2 += n.y_names
            c1 += [n.name for i in range(len(n.y_names))]
        cols = pd.MultiIndex.from_arrays([c1, c2])
        out = pd.DataFrame(None, 
                           index=[k for k in range(self.horizon)], 
                           columns=cols)
        
        xx = np.array([[1,2,3]]).T
        xx = xx.flatten()
        print(xx)
        out.loc[0, 'P1'] = xx

        print(out)

        print(out.loc[0, 'P1'])
        

        for k in range(self.horizon):
            print("=================")
            print("time step no. ", k)
            print("=================")
            for p in self.Ps:  # 1. simulate producers
                y = p.sim_step(k, self.flows)
                self.update_flows(y, p)
                out.loc[k, p.name] = y.flatten()
            for sc in self.SCs:  # 2. simulate SCs
                y = sc.sim_step(k, self.flows)
                self.update_flows(y, sc)
                out.loc[k, sc.name] = y.flatten()
            for c in self.Cs:
                y = c.sim_step(k, self.flows)
                print(out.loc[k, c.name])
                out.loc[k, c.name] = y.flatten()

        self.out = out
        if store: out.to_csv('results/out.csv')
    
