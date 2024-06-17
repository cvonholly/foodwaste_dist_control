import numpy as np

from data.data import get_P_facs
from nodes.node import Node


class P(Node):
    """
    class representing prodcuers
    """
    def __init__(self, name, T, a, 
                 flow_matrix: np.ndarray, 
                 flow_nodes: list,
                 x0: np.ndarray,
                 food_input) -> None:
        """
        inputs: 
            name
            T (numnber timesteps)
            a (factor) for factors 
            flow_matrix: matrix of output flow factors (array of size (n*T) x T)
            flow_nodes: nd.array of output nodes length k
            x0: initial state (array of size T)
            food_input: (list) food_input at time step k
        """
        self.name = name
        self.T = T
        self.a = a
        self.flow_matrix = flow_matrix   # output flow matrix
        self.flows_facs = np.diag(flow_matrix[:-1])  # output flow factors
        self.flow_nodes = flow_nodes  # output flow nodes (list of names)
        self.x0 = x0   # x0 state
        self.sz = x0.size   # size of state
        self.x = x0   # current state
        self.y = None   # output
        self.y_names = ['flow %s' % (i) for i in flow_nodes] + ['foodwaste', 'input flow']
        self.x_hist = []  # previous x's
        self.food_input = food_input
        self.alphas, self.facs_fw = get_P_facs(self.flows_facs, T-1, a)
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
        C = self.flow_matrix
        final_rows = np.zeros((2, C.shape[1]))
        final_rows[0, -1] = 1  # at final time step, everything hoes to waste
        final_rows[1, 0] = 1  # put food input also to output 
        C = np.vstack((
            C,
            final_rows))
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
        self.x = self.A @ self.x + self.B @ self.food_input[k]  # time step
        self.y = self.C @ self.x   # get output
        self.print_all()  # for debugging
        return self.y