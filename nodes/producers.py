import numpy as np
import cvxpy as cp

from nodes.node import Node
import control.mpc


class P(Node):
    """
    class representing prodcuers
    """
    def __init__(self, name, T,
                 flow_matrix: np.ndarray,
                 fw_matrix: np.ndarray,
                 flow_nodes: list,
                 x0: np.ndarray,
                 food_input,
                 ec_mpc=False,
                 q=None,
                 mpc_h=None) -> None:
        """
        inputs: 
            name
            T (numnber timesteps)
            flow_matrix: matrix of output flow factors (array of size (n*T) x T)
            fw_matrix: matrix row of food waste factors (size 1 x T)
            flow_nodes: nd.array of output nodes length k
            x0: initial state (array of size T)
            food_input: (list) food_input at time step k
            ec_mpc: (bool) weather to determin input via MPC
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
        # MPC inputs
        self.ec_mpc = ec_mpc
        self.total_food_input = np.sum(self.food_input)
        self.q = q
        self.mpc_h = mpc_h
    
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
        inp = control.mpc.mpc_P(self.A, self.B, self.q, self.x, self.total_food_input, self.mpc_h) if self.ec_mpc \
                else self.food_input[k]  # c
        if self.ec_mpc:
            self.total_food_input = np.round(self.total_food_input - inp, 8)  # have to adapt to input
            print("with local economic mpc the input was found to be: ", inp)
            print("without: ", self.food_input[k])
        self.x_hist.append(self.x)
        self.x = self.A @ self.x + self.B @ inp  # time step
        self.y = self.C @ self.x   # get output
        # self.print_all()  # for debugging
        return self.y
            