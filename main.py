import numpy as np

from nodes.producers import P
from nodes.consumers import C
from nodes.social_charity import SC
from simulation import Simulation


if __name__=="__main__":
    """
    parameters
    """
    T = 5
    alpha = .7
    beta = .3
    flows_fact = np.vstack([
        np.array([beta * np.exp(-.1*x) for x in range(T-1)]),  # f1
        np.array([beta * np.exp(-.1*x) for x in range(T-1)]),   # f2
        np.array([beta * np.exp(-.1*x) for x in range(T-1)])   # f3
    ])  # flow factors
    p1_out_flows = np.vstack([flows_fact[0], 
                              flows_fact[0],
                              flows_fact[1]])
    sc1_out_flows = np.vstack([flows_fact[2]])
    x0 = np.zeros((T, 1))  # x0
    horizon = 20   # simulation horizon
    input_flows = [[[1]] if k<=3 else [[0]] for k in range(horizon)]  # food input for horizon
    # input_flows = np.array(input_flows).T


    """
    nodes
    """
    P1 = P('P1', T, alpha, 
           p1_out_flows, 
           ["C1", 
            "C2", 
            "SC1"],  # output flow nodes
           x0, 
           input_flows)
    SC1 = SC('SC1', T, alpha, 
             np.ones(1), 
             sc1_out_flows, 
             ["C1"],
             x0)
    C1 = C('C1', T, alpha, 2, x0)
    C2 = C('C2', T, alpha, 1, x0)
    
    Ps = [P1]
    SCs = [SC1]
    Cs = [C1, C2]
    
    S = Simulation(horizon, Ps, SCs, Cs)
    S.simulate()

