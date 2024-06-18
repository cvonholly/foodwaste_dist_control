import numpy as np
import pandas as pd

from nodes.node import Node


class C(Node):
    """
    class representing consumers
    """
    def __init__(self, name, T, 
                 sc_matrix: np.ndarray,
                 fw_matrix: np.ndarray,
                 n_flows, 
                 x0) -> None:
        """
        inputs: 
            T (numnber timesteps)
            alpha (factor)
            n_flows (number input flows)
        """
        self.name = name
        self.T = T
        self.sc_matrix = sc_matrix
        self.fw_matrix = fw_matrix
        self.n_flows = n_flows
        self.x0 = x0   # x0 state
        self.x = x0   # current state
        self.y = None   # output
        self.x_hist = []  # previous x's
        self.y_names = ['self consumption', 'food waste']  # output names
        self.alphas = np.ndarray  # initialize alphas
        self.C = self.get_C()  # get output matrix
        if (self.alphas > 1).any():
            raise Exception("aborting, alpha value is greater 0")        
        self.A = self.get_A()
        self.B = None  # is computed at first time step (due to wait for input)
        
    
    # def get_A(self):
    #     """
    #     get standard system matrix from factors
    #     """
    #     # t = self.T-1
    #     # A = np.hstack((np.eye(t) - np.eye(t) * self.alphas, np.zeros([1,t]).T))
    #     # A = np.vstack((np.zeros(self.T), A))
    #     # return A
    #     return np.eye(self.T) - np.eye(self.T) * self.alphas
    
    def get_B(self, inputs: np.array):
        """
        inputs: input flows
            rows: nodes (n)
            cols: time step inputs (T)
        """
        self.n_u = inputs.shape  # = (n, T) = (#nodes, #states)
        self.B = np.hstack([np.eye(self.n_u[1]) for i in range(self.n_u[0])])  # hor. stacked identity matrices
        return self.B
    
    def get_C(self):
        C = np.vstack((self.sc_matrix, self.fw_matrix))
        self.alphas = C.sum(axis=0)  # alpha values for A matrix
        return C

    def sim_step(self, k, inputs: pd.DataFrame):
        inputs = inputs.fillna(0).to_numpy()  # inputs to numpy
        if self.B is None: self.B = self.get_B(inputs)  # computes B at first time step
        inputs = np.resize(inputs, (self.B.shape[1], 1))  # resize for matrix multiplication
        self.x_hist.append(self.x)
        self.x = self.A @ self.x + self.B @ inputs  # time step
        self.y = self.C @ self.x   # get output
        self.print_all()
        return self.y
        
