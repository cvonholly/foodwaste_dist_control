import numpy as np

from nodes.producers import P
from nodes.consumers import C
from nodes.social_charity import SC
from simulation import Simulation

from marks.flow_factors import flow_matrix_P_to_C, flow_matrix_P_to_SC, flow_matrix_SC_to_C, flow_matrix_C_consumption  # import marks computation methods


if __name__=="__main__":
    """
    parameters
    """
    T = 5  # food waste time horizon
    alpha = .7   # no idea
    beta = .8   # P's output flows: how much percent is flowing out (.8=80%)
    n_cs = 4
    n_scs = 3
    Cs_names = [f'C{i}' for i in range(n_cs)]   # set number consumers
    SCs_names  = [f'SC{i}' for i in range(n_scs)]   # set number sc's
        # flows_fact = np.vstack([
        #     np.array([np.exp(-.1*x) for x in range(T-1)]),  # f1: P to C's
        #     np.array([1/2 * np.exp(-.1*x) for x in range(T-1)]),   # f2: P to SC's
        #     np.array([1/3 * np.exp(-.1*x) for x in range(T-1)]),   # f3: SC to SC's
        # ])  # flow factors
    #
    # copied from marks test:
    food_waste = 0.08 # food waste percentage/ratio at producer level
    T_start = 3 # when producers start to give food products to charity
    T_end = 4 # when producers end to give food products to charity
    alpha_start = 0.25 # amount/percentage of food given to charity at beginning
    alpha_end = 0.5 # amount/percentage of food given to charity at end
    alpha_first_day = 0.2 # probability that food gets consumed from consumer on first day
    alpha_last_day = 0.8 # probability that food gets consumed from consumer on last day
    f0 = 1 #inflow to producer
    #
    #
    #
    matrix_P_to_C = flow_matrix_P_to_C(T, food_waste)
    matrix_P_to_SC = flow_matrix_P_to_SC(T, T_start, T_end, alpha_start, alpha_end)
    matrix_SC_to_C = flow_matrix_SC_to_C(T, T_start, T_end, alpha_start, alpha_end)

    p1_out_flows = np.vstack(
                            [np.vstack([matrix_P_to_C for i in range(len(Cs_names))]),
                             np.vstack([matrix_P_to_SC for i in range(len(SCs_names))]),
                            ]
    )
                            # before:
                            # [np.vstack([flows_fact[0] for i in range(len(Cs_names))]),
                            #   np.vstack([flows_fact[1] for i in range(len(SCs_names))])
                            # ]
    maxes = p1_out_flows.max(axis=0)
    maxes[np.where(maxes==0)] = 1  # make sure not to devide by 0              
    p1_out_flows = beta * p1_out_flows / (maxes * p1_out_flows.shape[0])  # normnalize martix w.r.t. beta
    
    sc1_out_flows = np.vstack([matrix_SC_to_C for i in range(len(Cs_names))])
    maxes = sc1_out_flows.max(axis=0)
    maxes[np.where(maxes==0)] = 1  # make sure not to devide by 0
    sc1_out_flows = beta * sc1_out_flows / (maxes * p1_out_flows.shape[0])  # normnalize martix w.r.t. beta

    x0 = np.zeros((T, 1))  # x0
    horizon = 20   # simulation horizon: has to be sufficiently longer then T for simulation to be correct !
    input_flows = [[[1]] if k<=3 else [[0]] for k in range(horizon)]  # food input for horizon


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
             sc1_out_flows, 
             Cs_names,
             x0))
    Cs = []
    for cc in Cs_names:
        Cs.append(C(cc, T, alpha, len(Ps)+len(SCs), x0))
    
    
    
    S = Simulation(horizon, 
                   T,
                   Ps, SCs, Cs)
    S.simulate()

