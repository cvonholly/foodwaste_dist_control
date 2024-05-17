import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from plotting import plots

def get_facs(T, a):
    """
    gets factors of consumption and foodwaste

    todo: extend me beyond linear model

    returns:
        gammas (for a matrix)
        self consumption factors
        foodwaste factors
    """
    foodwaste = np.linspace(0, a, T)
    self_con = np.flip(foodwaste)
    if any(foodwaste+self_con > 1):
        print("aborting, invalid factors for c class")
        return False, False
    return foodwaste+self_con, self_con, foodwaste


class C:
    """
    class representing consumers
    """
    def __init__(self, T, alpha, n_flows, x0) -> None:
        """
        inputs: 
            T (numnber timesteps)
            alpha (factor)
            n_flows (number input flows)
        """
        self.T = T
        self.alpha = alpha
        self.n_flows = n_flows
        self.x0 = x0   # x0 state
        self.x = x0   # current state
        self.y = None   # output
        self.x_hist = []  # previous x's
        self.gammas, self.facs_sc, self.facs_fw = get_facs(T-1, alpha)
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

    def sim_step(self, flows):
        if len(flows)!=self.n_flows:
            print("flows are not inputed correctly, aborting")
            return
        self.x_hist.append(self.x)
        self.y = self.C @ self.x   # get output
        self.x = self.A @ self.x + self.B @ flows  # time step
        return self.y
        

    def print(self):
        print("x0:")
        print(self.x0)
        print("A: ")
        print(self.A)
        print("B: ")
        print(self.B)
        print("C: ")
        print(self.C)


def get_Fs(k):
    if k<=3:
        return np.ones([2,1])
    else:
        return np.zeros([2,1])

class Simulation:
    def __init__(self, Cs, Ps, SCs):
        """
        inputs:
            Cs: list of consumers
            Ps: list of producers
            SCs: list of sharers
        """
        self.Cs = Cs
        self.Ps = Ps
        self.SCs = SCs

    def sim_step(self, K=10, pprint=True, plot=True):
        """
        todo: include Ps and SCs
        """
        df_out = pd.DataFrame(data=None,
                           columns=['self_con',
                                    'foodwaste',
                                    'input'],
                                    index=[i for i in range(K)])
        out = []
        for k in range(K):
            fs = get_Fs(k)
            for C in self.Cs:
                C.sim_step(fs)
                # out = np.append(out,np.vstack([C.y, 
                #                       C.x,
                #                       fs.sum()]))
                out.append(np.vstack([C.y, 
                                      C.x,
                                      fs.sum()]))

                
        
        # check for energy conservation
        # df_out = pd.DataFrame(out)
        
        # print(df_out)
        
        if pprint: print(out)
        if plot: plots(out)
