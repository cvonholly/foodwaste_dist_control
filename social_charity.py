import numpy as np
import pandas as pd

from data import get_SC_facs
from node import Node


class SC(Node):
    """
    class representing social and charity actors
    """
    def __init__(self, name, T, a, 
                 in_flows_facs,
                out_flows_facs, 
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
        self.in_flows_facs = in_flows_facs  # input flow factors. generally 1 x flow
        self.n_u = in_flows_facs.shape[0]
        self.out_flows_facs = out_flows_facs   # output flow factors
        self.flow_nodes = flow_nodes
        self.x0 = x0   # x0 state
        self.sz = x0.size   # size of state
        self.x = x0   # current state
        self.y = None   # output
        self.x_hist = []  # previous x's
        self.alphas, self.facs_fw = get_SC_facs(out_flows_facs, T-1, a)
        if (self.alphas > 1).any():
            raise Exception("aborting, alpha value is greater 0")
        self.A = self.get_A()
        self.B = self.get_B()
        self.C = self.get_C()
    
    def get_A(self):
        """
        get standard system matrix from factors
        """
        t = self.T-1
        y = np.hstack((np.eye(t) - np.eye(t) * self.alphas, np.zeros([1,t]).T))
        y = np.vstack((np.zeros(self.T), y))
        return y
    
    def get_B(self):
        """
        input matrix
            first row: input flow factors
            other rows: zeros
        """
        return np.vstack((self.in_flows_facs, 
                          np.zeros((self.sz-1, self.n_u))))
    
    def get_C(self):
        C = np.vstack((
            self.out_flows_facs,
            self.facs_fw))
        zz = np.zeros((C.shape[0], 1))
        zz[-1] = 1  # at final time, everything goes to waste
        C = np.hstack((C, zz))
        return C

    def sim_step(self, k: int, inputs: pd.DataFrame):
        """
        k: time step k
        flows: n x n matrix of flows

        return: y (output consisting of)
            flows: np.array of output flows
            store: float represnting stored amount at time step t
            foodwaste: float represnting foodwaste at time step t
        """
        inputs = [inputs[inputs[self.name].notna()][self.name].to_numpy()]
        self.x_hist.append(self.x)
        self.y = self.C @ self.x   # get output
        self.x = self.A @ self.x + self.B @ inputs  # time step
        print("inputs: ")
        print(inputs)
        self.print_all()
        return self.y