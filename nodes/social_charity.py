import numpy as np
import pandas as pd

from nodes.node import Node


class SC(Node):
    """
    class representing social and charity actors
    """
    def __init__(self, name, T,
                out_flows: np.ndarray,
                fw_matrix: np.ndarray,
                flow_nodes: list,
                x0: np.ndarray) -> None:
        """
        inputs: 
            name
            T (number timesteps)
            out_flows: np.ndarray,
            fw_matrix: np.ndarray,
            flow_nodes: output flow nodes names (list)
            x0: initial state (array of size T)
        """
        self.name = name
        self.T = T
        self.out_flows = out_flows   # output flow matrix (n*T) x T
        self.n = int(out_flows.shape[0]/T)   # number outflows
        self.fw_matrix = fw_matrix   # food waste matrix
        self.flow_nodes = flow_nodes
        self.x0 = x0   # x0 state
        self.sz = x0.size   # size of state = T
        self.x = x0   # current state
        self.y = None   # output
        self.y_names = ['flow %s' % (i+1) for i in range(self.n)] + ['foodwaste']
        self.x_hist = []  # previous x's
        self.alphas = np.ndarray
        self.C = self.get_C()
        if (self.alphas > 1).any():
            raise Exception("aborting, alpha value is greater 0")
        self.A = self.get_A()
        self.B = None
    
    # def get_A(self):
    #     """
    #     get standard system matrix from factors
    #     """
    #     # t = self.T-1
    #     # y = np.hstack((np.eye(t) - np.eye(t) * self.alphas, np.zeros([1,t]).T))
    #     # y = np.vstack((np.zeros(self.T), y))
    #     # return y
    #     return np.eye(self.T) - np.eye(self.T) * self.alphas
    
    def get_B(self, inputs: np.ndarray):
        """
        inputs: input flows
            rows: nodes (n)
            cols: time step inputs (T)
        """
        self.n_u = inputs.shape  # = (n, T) = (#nodes, #states)
        self.B = np.hstack([np.eye(self.n_u[1]) for i in range(self.n_u[0])])  # horizontal stacked identity matrices
        return self.B
    
    def get_C(self):
        C = np.vstack((self.out_flows, self.fw_matrix))
        self.alphas = C.sum(axis=0)  # alpha values for A matrix
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
        inputs = inputs.fillna(0).to_numpy()  # inputs to numpy
        if self.B is None: self.B = self.get_B(inputs)  # computes B at first time step
        inputs = np.resize(inputs, (self.B.shape[1], 1))
        self.x_hist.append(self.x)
        self.x = self.A @ self.x + self.B @ inputs  # time step
        self.y = self.C @ self.x   # get output
        self.print_all()
        return self.y