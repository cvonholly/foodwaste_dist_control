import numpy as np
import pandas as pd
import networkx as nx

from nodes.producers import P
from nodes.consumers import C
from nodes.social_charity import SC
from nodes.node import Node
    

class Simulation:
    """
    simulation graph

    nodes = list of all nodes ordered by Ps, SCs, Cs
    flows = n x n matrix with np.NaN 
    """
    def __init__(self, 
                 horizon: int,
                 state_size: int,
                 Ps: list[P], SCs: list[SC], Cs: list[C]):
        """
        inputs:
            horizon: simulation horizon
            state_size: size of state
            Ps: list of producers
            SCs: list of sharers
            Cs: list of consumers
        """
        self.horizon = horizon
        self.state_size = state_size
        self.Ps = Ps
        self.SCs = SCs
        self.Cs = Cs
        self.nodes = Ps + SCs + Cs
        self.names = [n.name for n in self.nodes]
        self.n_nodes = len(self.nodes)
        # flows have to be adapted: it is relevant how old the flows are between the time steps. we create a new matric for that
        idx_t = pd.MultiIndex.from_product((self.names, list(range(state_size))))
        self.flows_t = pd.DataFrame(np.NAN, index=self.names, columns=idx_t)  # flows in graph with time element
        self.flows = pd.DataFrame(np.NAN, index=self.names, columns=self.names)  # flows in graph summed
        self.out = self.get_out_df()
        self.all_flows = pd.DataFrame(pd.NA, 
                                      columns=pd.MultiIndex.from_product(([t for t in range(self.horizon)], self.names)),
                                      index=self.names)
    
    def update_flows(self, y: np.ndarray, N: Node, k: int):
        """
        update flows and output
        k: simulation time step
        """
        out_last = np.array([])
        if type(N)==P: 
            out_last = list(y[-2:].flatten())
            y = y[:-2]  # exclude last rows (foodwaste) to get all flows
        elif type(N)==SC: 
            out_last = list(y[-1:].flatten())
            y = y[:-1]   # exclude final row
        else:
            raise Exception('invalid Node type')

        # todo: check has to be adapted
        # if len(N.flow_nodes)!=y.shape[0]:  # check for errors
        #     print(N.flow_nodes)
        #     print(y)
        #     raise Exception("idxs shape does not match y shape")
        
        print("updating flows for ", N.name)
        out = []
        for i in range(len(N.flow_nodes)):  # iterate over output nodes by index
            # for x in range(self.state_size):  # iterate over time / state
            y_i = y[i*self.state_size : (i+1)*self.state_size].T
            print("y_i: ----")
            print(y_i)
            print("y_i.sum(): ----")
            print(y_i.sum())
            print("self.flows_t.loc[%s, %s]: ----" % (N.name, N.flow_nodes[i]))
            print(self.flows_t.loc[N.name, N.flow_nodes[i]])
            self.flows_t.loc[N.name, N.flow_nodes[i]] = y_i  # update time dependent flow
            self.flows.loc[N.name, N.flow_nodes[i]] = y_i.sum()   # update final flow
            out.append(y_i.sum())
        # out = self.flows.loc[N.name, :].values + out_last
        out = out + out_last
        print("out: ---")
        print(out)
        print('self.out.loc[k, N.name]: ---')
        print(self.out.loc[k, N.name])
        self.out.loc[k, N.name] = out

    def get_out_df(self) -> pd.DataFrame:
        c1, c2 = [], []
        for n in self.nodes:
            c2 += n.y_names
            c1 += [n.name for i in range(len(n.y_names))]
        cols = pd.MultiIndex.from_arrays([c1, c2])
        return pd.DataFrame(None, 
                           index=[k for k in range(self.horizon)], 
                           columns=cols)


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
        for k in range(self.horizon):
            print("=================")
            print("time step no. ", k)
            print("=================")
            for p in self.Ps:  # 1. simulate producers
                y = p.sim_step(k, self.flows_t[p.name])
                self.update_flows(y, p, k)  
                    # updates summed and total flow
                    # and update output
            for sc in self.SCs:  # 2. simulate SCs
                y = sc.sim_step(k, self.flows_t[sc.name])
                self.update_flows(y, sc, k)  
                    # updates summed anf total flow
                    # and update output
            for c in self.Cs:
                y = c.sim_step(k, self.flows_t[c.name])
                self.out.loc[k, c.name] = y.flatten()
            if store:
                self.all_flows[k] = self.flows

        if store:
            self.out.to_csv('results/out.csv')
            self.all_flows.to_csv('results/flows.csv')

    
