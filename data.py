import numpy as np


def asymp_facs(T, a):
    """
    generates asymp. increasing factors until 1
    """
    out = [np.exp(a*(i-T)) for i in range(1,T+1)]
    return out


# properties of c1
class C:
    def __init__(self, T, alpha, n_flows) -> None:
        self.T = T
        self.alpha = alpha
        self.n_flows = n_flows
        self.facs = asymp_facs(T-1, alpha)
        self.A = self.get_A_from_facs()
        self.B = self.get_B()
    
    def get_A_from_facs(self):
        """
        get standard system matrix from factors
        """
        t = len(self.facs)
        y = np.hstack((np.eye(t) - np.eye(t) * self.facs, np.zeros([1,t]).T))
        y = np.vstack((np.zeros(self.T), y))
        return y
    
    def get_B(self):
        ones = np.ones(self.n_flows)
        zeros = np.zeros((self.T-1, self.n_flows))
        return np.vstack((ones,zeros))

    def print(self):
        print("A: ")
        print(self.A)
        print("B: ")
        print(self.B)