import numpy as np

from .data import params, base


def load_params(name: str) -> dict:
    p = params[name]
    # load params
    if 'greedy' not in p:
        p['greedy'] = False
    p["Ps_names"] = [f'P{i}' for i in range(p["n_ps"])] 
    p["Cs_names"] = [f'C{i}' for i in range(p["n_cs"])] 
    p["SCs_names"] = [f'SC{i}' for i in range(p["n_scs"])]
    p["x0_p"] = np.array([base['nv_p_d'] * p['n_cs'] * p['fw_model']['store']**(i) for i in range(p["T"])], ndmin=2).T  # prodcuers initial state: like if they have simulated steady state
    if p["inp_params"][0]=='base case': 
        # in this case, the input flow is just the daily requirement of the consumers
        p["input_flows"] = [[[base['nv_p_d'] * p['n_cs'] / p['n_ps']]]   if k<=p["inp_params"][1] else [[0]] for k in range(p["horizon"])]  # food input for horizon]
    else:
        p["input_flows"] = [[[p["inp_params"][0]]] if k<=p["inp_params"][1] else [[0]] for k in range(p["horizon"])]  # food input for horizon
    return p
