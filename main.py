import numpy as np

from producers import P
from consumers import C
from social_charity import SC
from flow import Flow
from simulation import Simulation


if __name__=="__main__":
    T = 5   # parameters
    alpha = .2
    flows_fact = np.vstack([
        np.array([.1 * np.exp(-.1*x) for x in range(T-1)]),  # f1
        np.array([.1 * np.exp(-.1*x) for x in range(T-1)]),   # f2
        np.array([.1 * np.exp(-.1*x) for x in range(T-1)])   # f3
    ])  # flow factors
    p1_out_flows = np.vstack([flows_fact[0], flows_fact[1]])

    x0 = np.zeros((1,T))  # x0

    horizon = 10   # simulation horizon
    input_flows = [1 if k<=3 else 0 for k in range(horizon)]  # food input for horizon

    P1 = P('P1', T, alpha, p1_out_flows, x0, input_flows)
    C1 = C('C1', T, alpha, 2, x0)
    SC1 = SC('SC1', T, alpha, np.ones(1), flows_fact[2], x0)
    Ps = [P1]
    Cs = [C1]
    SCs = [SC1]
    all_nodes = Ps
    all_nodes.extend(Cs)
    all_nodes.extend(SCs)
    n_nodes = len(all_nodes)
    
    # flows adj. list
    flows_list = [Flow('f1', P1.name, C1.name, 0),
                  Flow('f2', P1.name, SC1.name, 0),
                  Flow('f3', SC1.name, C1.name, 0)]
    # flows = [[0, flows_list[0], flows_list[1]],
    #          [0 for i in range(n_nodes)],
    #          [0, flows_list[2], 0]]

    S = Simulation(horizon, Ps, Cs, SCs,
                   n_nodes, all_nodes, flows_list)
    S.sim_step()
