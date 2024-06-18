import numpy as np

from nodes.producers import P
from nodes.consumers import C
from nodes.social_charity import SC
from simulation import Simulation

from marks.flow_factors import flow_matrix_P_to_C, flow_matrix_P_to_SC, flow_matrix_SC_to_C, flow_matrix_C_consumption  # import marks computation methods
from marks.foodwaste_matrices import *


if __name__=="__main__":
    """
    parameters
    """
    horizon = 10   # simulation horizon: has to be sufficiently longer then T for simulation to be correct !
    T = 5  # food waste time horizon
    n_ps = 1   # set number producers
    n_cs = 4  # set number consumers
    n_scs = 3   # set number social-charities
    Ps_names = [f'P{i}' for i in range(n_ps)] 
    Cs_names = [f'C{i}' for i in range(n_cs)] 
    SCs_names  = [f'SC{i}' for i in range(n_scs)]
    x0 = np.zeros((T, 1))  # initial state
    input_flows = [[[1]] if k<=3 else [[0]] for k in range(horizon)]  # food input for horizon
    # copied from marks test:
    food_waste = 0.08 # food waste percentage/ratio at producer level
    T_start = 3 # when producers start to give food products to charity
    T_end = 4 # when producers end to give food products to charity
    alpha_start = 0.25 # amount/percentage of food given to charity at beginning
    alpha_end = 0.5 # amount/percentage of food given to charity at end
    alpha_first_day = 0.2 # probability that food gets consumed from consumer on first day
    alpha_last_day = 0.8 # probability that food gets consumed from consumer on last day
    f0 = 1 #inflow to producer

    # flow matrices
    matrix_P_to_C = flow_matrix_P_to_C(T, food_waste)
    matrix_P_to_SC = flow_matrix_P_to_SC(T, T_start, T_end, alpha_start, alpha_end)
    matrix_SC_to_C = flow_matrix_SC_to_C(T, T_start, T_end, alpha_start, alpha_end)
    matrix_C_consumption = flow_matrix_C_consumption(T, alpha_first_day, alpha_last_day)

    # foodwaste matrices
    p_fw_matrix = get_simple_fw_matrix(matrix_P_to_C)
    c_sc_matrix = matrix_C_consumption  # how much self consumption of consumers
    sc_fw_matrix, c_fw_matrix = p_fw_matrix, p_fw_matrix  # simple model

    # compute producer output flow
    p_out_flows = np.vstack([np.vstack([matrix_P_to_C for i in range(len(Cs_names))]) / n_cs,
                             np.vstack([matrix_P_to_SC for i in range(len(SCs_names))]) / n_scs,
                            ])
    
    # compute sc out flows
    sc_out_flows = np.vstack([matrix_SC_to_C for i in range(len(Cs_names))]) / n_cs    


    """
    nodes
    """
    Ps = []
    for p in Ps_names:
        Ps.append(P(p, T, 
           p_out_flows,
           p_fw_matrix,
           Cs_names + SCs_names,  # output flow nodes
           x0, 
           input_flows))    
    SCs = []
    for sc in SCs_names:
        SCs.append(SC(sc, T,
             sc_out_flows,
             sc_fw_matrix,
             Cs_names,
             x0))
    Cs = []
    for cc in Cs_names:
        Cs.append(C(cc, T,
            c_sc_matrix,
            c_fw_matrix,
            len(Ps)+len(SCs), 
            x0))
    
    
    
    S = Simulation(horizon, 
                   T,
                   Ps, SCs, Cs)
    S.simulate()

