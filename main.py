import numpy as np


from params.params import load_params
from nodes.producers import P
from nodes.consumers import C
from nodes.social_charity import SC
from simulation import Simulation
from marks.flow_factors import flow_matrix_P_to_C, flow_matrix_P_to_SC, flow_matrix_SC_to_C, flow_matrix_C_consumption  # import marks computation methods
from marks.foodwaste_matrices import *


if __name__=="__main__":
    # load parameters
    name = "EC_MPC"
    params = load_params(name)
    print(params)
    T = params["T"]  # food waste time horizon
    n_ps = params["n_ps"] # set number producers
    n_cs = params["n_cs"]  # set number consumers
    n_scs = params["n_scs"]   # set number social-charities
    Ps_names = params["Ps_names"]
    Cs_names = params["Cs_names"]
    SCs_names  = params["SCs_names"]

    # flow matrices
    matrix_P_to_C = flow_matrix_P_to_C(T, params["food_waste"])
    matrix_P_to_SC = flow_matrix_P_to_SC(T, params["T_start"], params["T_end"], 
                                         params["alpha_start"], params["alpha_end"])
    matrix_SC_to_C = flow_matrix_SC_to_C(T, params["T_start"], params["T_end"], 
                                         params["alpha_start"], params["alpha_end"])
    matrix_C_consumption = flow_matrix_C_consumption(T, params["alpha_first_day"], params["alpha_last_day"])

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
           params["x0"], 
           params["input_flows"],
           ec_mpc=params["ec_mpc"],
           q=p_fw_matrix.T,
           mpc_h=params["mpc_h"]))    
    SCs = []
    for sc in SCs_names:
        SCs.append(SC(sc, T,
             sc_out_flows,
             sc_fw_matrix,
             Cs_names,
             params["x0"]))
    Cs = []
    for cc in Cs_names:
        Cs.append(C(cc, T,
            # c_sc_matrix,
            c_fw_matrix,
            len(Ps)+len(SCs), 
            params["x0"],
            ec_mpc=params["ec_mpc_c"],
            food_intake=params["food_intake"],
            mpc_h=params["mpc_h_c"]))
    
    
    
    S = Simulation(name,
                    params["horizon"], 
                   T,
                   Ps, SCs, Cs)
    S.simulate()

