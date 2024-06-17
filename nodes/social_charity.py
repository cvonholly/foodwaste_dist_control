import numpy as np
import pandas as pd

from data.data import get_SC_facs
from nodes.node import Node


class SC(Node):
    """
    class representing social and charity actors
    """
    def __init__(self, name, T, a, 
                out_flows, 
                flow_nodes,
                x0: np.ndarray) -> None:
        """
        inputs: 
            name
            T (numnber timesteps)
            a (factor) for factors 
            flows factors: matrix of output flow factors (array of size n x T)
            x0: initial state (array of size T)
        """
        self.name = name
        self.T = T
        self.a = a
        self.out_flows = out_flows   # output flow matrix (n*T) x T
        self.n = int(out_flows.shape[0]/T)   # number outflows
        self.flows_facs = np.diag(out_flows[:-1])  # output flow factors
          # todo: this assumes we have same flow factors for every output connection
          #       needs to be adapted to multiple different output flows at model
          #       is extended
        self.flow_nodes = flow_nodes
        self.x0 = x0   # x0 state
        self.sz = x0.size   # size of state = T
        self.x = x0   # current state
        self.y = None   # output
        self.y_names = ['flow %s' % (i+1) for i in range(self.n)] + ['foodwaste']
        self.x_hist = []  # previous x's
        self.alphas, self.facs_fw = get_SC_facs(self.flows_facs, T-1, a)
        if (self.alphas > 1).any():
            raise Exception("aborting, alpha value is greater 0")
        self.A = self.get_A()
        self.B = None
        self.C = self.get_C()
    
    def get_A(self):
        """
        get standard system matrix from factors
        """
        t = self.T-1
        y = np.hstack((np.eye(t) - np.eye(t) * self.alphas, np.zeros([1,t]).T))
        y = np.vstack((np.zeros(self.T), y))
        return y
    
    def get_B(self, inputs: np.array):
        """
        inputs: input flows
            rows: nodes (n)
            cols: time step inputs (T)
        """
        self.n_u = inputs.shape  # = (n, T) = (#nodes, #states)
        self.B = np.hstack([np.eye(self.n_u[1]) for i in range(self.n_u[0])])  # horizontal stacked identity matrices
        return self.B
    
    def get_C(self):
        C = self.out_flows
        final_row = np.zeros((1, C.shape[1]))
        final_row[0, -1] = 1 # at final time step, everything hoes to waste
        C = np.vstack((
            C,
            final_row))
        return C
        # old:
        # C = np.vstack((
        #     self.out_flows_facs,
        #     self.facs_fw))
        # zz = np.zeros((C.shape[0], 1))
        # zz[-1] = 1  # at final time, everything goes to waste
        # C = np.hstack((C, zz))
        # return C

    def sim_step(self, k: int, inputs: pd.DataFrame):
        """
        k: time step k
        flows: n x n matrix of flows

        return: y (output consisting of)
            flows: np.array of output flows
            store: float represnting stored amount at time step t
            foodwaste: float represnting foodwaste at time step t
        """
        # inputs = [inputs[inputs[self.name].notna()][self.name].to_numpy()]  # old
        inputs = inputs.fillna(0).to_numpy()  # inputs to numpy
        if self.B is None: self.B = self.get_B(inputs)  # computes B at first time step
        inputs = np.resize(inputs, (self.B.shape[1], 1))
        self.x_hist.append(self.x)
        self.x = self.A @ self.x + self.B @ inputs  # time step
        self.y = self.C @ self.x   # get output
        self.print_all()
        return self.y