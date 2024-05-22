import numpy as np

from data import get_P_facs
from node import Node


class P(Node):
    """
    class representing prodcuers
    """
    def __init__(self, name, T, a, 
                 flows_facs: np.ndarray, 
                 flow_nodes: list,
                 x0: np.ndarray,
                 food_input) -> None:
        """
        inputs: 
            name
            T (numnber timesteps)
            a (factor) for factors 
            flows factors: matrix of output flow factors (array of size k x T)
            flow_nodes: nd.array of output nodes length k
            x0: initial state (array of size T)
            food_input: (list) food_input at time step k
        """
        self.name = name
        self.T = T
        self.a = a
        self.flows_facs = flows_facs  # output flow factors
        self.flow_nodes = flow_nodes  # output flow nodes (list of names)
        self.x0 = x0   # x0 state
        self.sz = x0.size   # size of state
        self.x = x0   # current state
        self.y = None   # output
        self.y_names = ['flow %s' % (i+1) for i in range(flows_facs.shape[0])] + ['foodwaste']
        self.x_hist = []  # previous x's
        self.food_input = food_input
        self.alphas, self.facs_fw = get_P_facs(flows_facs, T-1, a)
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
        """
        return np.vstack((1, np.zeros((self.sz-1,1))))
    
    def get_C(self):
        C = np.vstack((
            self.flows_facs, 
            self.facs_fw))
        zz = np.zeros((C.shape[0], 1))
        zz[-1] = 1  # at final time, everything goes to waste
        C = np.hstack((C, zz))
        return C

    def sim_step(self, k, flows):
        """
        k: time step k
        flows: not used

        return: y (output consisting of)
            flows: np.array of output flows
            store: float represnting stored amount at time step t
            foodwaste: float represnting foodwaste at time step t
        """
        self.x_hist.append(self.x)
        self.y = self.C @ self.x   # get output
        self.x = self.A @ self.x + self.B @ self.food_input[k]  # time step
        self.print_all()
        return self.y