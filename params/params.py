import numpy as np

from .data import params, base


def load_params(name: str) -> dict:
    p = params[name]
    # load params
    p["Ps_names"] = [f'P{i}' for i in range(p["n_ps"])] 
    p["Cs_names"] = [f'C{i}' for i in range(p["n_cs"])] 
    p["SCs_names"] = [f'SC{i}' for i in range(p["n_scs"])]
    if p["inp_params"][0]=='base case':
        p["input_flows"] = [[[base['nv_p_d']*p['n_cs']/p["inp_params"][1]]]  if k<=p["inp_params"][1] else [[0]] for k in range(p["horizon"])]  # food input for horizon]
    else:
        p["input_flows"] = [[[p["inp_params"][0]]] if k<=p["inp_params"][1] else [[0]] for k in range(p["horizon"])]  # food input for horizon
    return p