import numpy as np

from nodes.producers import P
from nodes.consumers import C
from nodes.social_charity import SC
from simulation import Simulation


if __name__=="__main__":
    """
    parameters
    """
    T = 5  # food waste time horizon
    alpha = .7   # no idea
    beta = .8   # P's output flows: how much percent is flowing out (.8=80%)
    Cs_names = [f'C{i}' for i in range(6)]   # set number consumers
    SCs_names  = [f'SC{i}' for i in range(3)]   # set number consumers
    flows_fact = np.vstack([
        np.array([np.exp(-.1*x) for x in range(T-1)]),  # f1: P to C's
        np.array([1/2 * np.exp(-.1*x) for x in range(T-1)]),   # f2: P to SC's
        np.array([1/3 * np.exp(-.1*x) for x in range(T-1)]),   # f3: SC to SC's
    ])  # flow factors
    p1_out_flows = np.vstack([np.vstack([flows_fact[0] for i in range(len(Cs_names))]),
                              np.vstack([flows_fact[1] for i in range(len(SCs_names))])
                            ])
    p1_out_flows = beta * p1_out_flows / (p1_out_flows.max(axis=0) * p1_out_flows.shape[0])  # normnalize martix w.r.t. beta
    sc1_out_flows = np.vstack([flows_fact[2] for i in range(len(Cs_names))])

    sc1_out_flows = beta * sc1_out_flows / (sc1_out_flows.max(axis=0) * p1_out_flows.shape[0])  # normnalize martix w.r.t. beta

    x0 = np.zeros((T, 1))  # x0
    horizon = 20   # simulation horizon: has to be sufficiently longer then T for simulation to be correct !
    input_flows = [[[1]] if k<=3 else [[0]] for k in range(horizon)]  # food input for horizon
    # input_flows = np.array(input_flows).T

    print(p1_out_flows)


    """
    nodes
    """
    P1 = P('P1', T, alpha, 
           p1_out_flows, 
           Cs_names + SCs_names,  # output flow nodes
           x0, 
           input_flows)
    Ps = [P1]
    SCs = []
    for sc in SCs_names:
        SCs.append(SC(sc, T, alpha, 
             np.ones(1), 
             sc1_out_flows, 
             Cs_names,
             x0))
    Cs = []
    for cc in Cs_names:
        Cs.append(C(cc, T, alpha, len(Ps)+len(SCs), x0))
    
    
    
    S = Simulation(horizon, Ps, SCs, Cs)
    S.simulate()

