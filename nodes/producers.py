import numpy as np

from nodes.node import Node


class P(Node):
    """
    class representing prodcuers
    """
    def __init__(self, name, T,
                 flow_matrix: np.ndarray,
                 fw_matrix: np.ndarray,
                 flow_nodes: list,
                 x0: np.ndarray,
                 food_input) -> None:
        """
        inputs: 
            name
            T (numnber timesteps)
            flow_matrix: matrix of output flow factors (array of size (n*T) x T)
            fw_matrix: matrix row of food waste factors (size 1 x T)
            flow_nodes: nd.array of output nodes length k
            x0: initial state (array of size T)
            food_input: (list) food_input at time step k
        """
        self.name = name
        self.T = T
        self.flow_matrix = flow_matrix   # output flow matrix
        self.fw_matrix = fw_matrix   # food waste matrix
        self.flow_nodes = flow_nodes  # output flow nodes (list of names)
        self.x0 = x0   # x0 state
        self.sz = x0.size   # size of state
        self.x = x0   # current state
        self.y = None   # output
        self.y_names = ['flow %s' % (i) for i in flow_nodes] + ['foodwaste', 'input flow']
        self.x_hist = []  # previous x's
        self.food_input = food_input
        self.alphas = np.ndarray
        self.C, self.C_bal = self.get_C()
        if (self.alphas > 1).any():
            raise Exception("aborting, alpha value is greater 0")
        self.A = self.get_A()
        self.B = self.get_B()

    
    # def get_A(self):
    #     """
    #     get standard system matrix from factors
    #     """
    #     # t = self.T-1
    #     # y = np.hstack((np.eye(t) - np.eye(t) * self.alphas, np.zeros([1,t]).T))
    #     # y = np.vstack((np.zeros(self.T), y))
    #     return np.eye(self.T) - np.eye(self.T) * self.alphas
    
    def get_B(self):
        """
        input matrix: put food input into x(t=0)
        """
        return np.vstack((1, np.zeros((self.sz-1,1))))
    
    def get_C(self):
        C_bal = np.vstack((self.flow_matrix, self.fw_matrix))   # balanced output matrix
        self.alphas = C_bal.sum(axis=0)  # alpha values for A matrix
        final_row = np.zeros((1, C_bal.shape[1]))
        final_row[0, 0] = 1  # put food input also to output 
        C = np.vstack((
            C_bal,
            final_row))
        return C, C_bal

    def sim_step(self, k, flows):
        """
        k: time step k
        flows: not used

        return: y (output consisting of)
            flows: np.array of output flows (n*T size)
            foodwaste: foodwaste at time step t
            input: input at t
        """
        self.x_hist.append(self.x)
        self.x = self.A @ self.x + self.B @ self.food_input[k]  # time step
        self.y = self.C @ self.x   # get output
        self.print_all()  # for debugging
        return self.y