import numpy as np


class Node:
    """
    base class for node
    """
    def __init__(self, name):
        self.name = name
        self.x0 = np.array([])
        self.x_hist = []
        self.y = np.array([])  # output
        self.y_names = []   # output names
        self.A = np.matrix([])
        self.B = np.matrix([])
        self.C = np.matrix([])
        self.D = np.matrix([])
        self.flow_nodes = list[str]   # output flow nodes
    
    def sim_step(self, t: int, input: np.matrix):
        """
        simulate time step
        input:
            t: int for time step
            input: np.matrix of flows
        """
        self.x_hist.append(self.x)
        self.y = self.C @ self.x   # get output
        self.x = self.A @ self.x + self.B @ input  # time step
        return self.y
    
    def print_all(self):
        print("printing all for node ", self.name)
        print("-----------------------------")
        print("x:")
        print(self.x)
        print("A: ")
        print(self.A)
        print("B: ")
        print(self.B)
        print("C: ")
        print(self.C)
        print("y: ")
        print(self.y)
