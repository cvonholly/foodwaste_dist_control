import numpy as np
import pandas as pd

from data import get_facs
from node import Node


class C(Node):
    """
    class representing consumers
    """
    def __init__(self, name, T, alpha, n_flows, x0) -> None:
        """
        inputs: 
            T (numnber timesteps)
            alpha (factor)
            n_flows (number input flows)
        """
        self.name = name
        self.T = T
        self.alpha = alpha
        self.n_flows = n_flows
        self.x0 = x0   # x0 state
        self.x = x0   # current state
        self.y = None   # output
        self.x_hist = []  # previous x's
        self.gammas, self.facs_sc, self.facs_fw = get_facs(T-1, alpha)
        if (self.gammas > 1).any():
            raise Exception("aborting, alpha value is greater 0")
        self.y_names = ['self consumption', 'food waste']
        self.A = self.get_A()
        self.B = self.get_B()
        self.C = self.get_C()
    
    def get_A(self):
        """
        get standard system matrix from factors
        """
        t = self.T-1
        y = np.hstack((np.eye(t) - np.eye(t) * self.gammas, np.zeros([1,t]).T))
        y = np.vstack((np.zeros(self.T), y))
        return y
    
    def get_B(self):
        ones = np.ones(self.n_flows)
        zeros = np.zeros((self.T-1, self.n_flows))
        return np.vstack((ones,zeros))
    
    def get_C(self):
        return np.hstack(((np.vstack((self.facs_sc, 
                          self.facs_fw))),
                          np.array([[0],[1]])))

    def sim_step(self, k, inputs: pd.DataFrame):
        inputs = np.array([inputs[inputs[self.name].notna()][self.name].to_numpy()]).T
        self.x_hist.append(self.x)
        self.y = self.C @ self.x   # get output
        self.x = self.A @ self.x + self.B @ inputs  # time step
        return self.y
        
