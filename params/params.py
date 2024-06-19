import numpy as np

from .data import params

print(params)



def load_params(name: str) -> dict:
    p = params[name]
    # load params
    p["Ps_names"] = [f'P{i}' for i in range(p["n_ps"])] 
    p["Cs_names"] = [f'C{i}' for i in range(p["n_cs"])] 
    p["SCs_names"] = [f'SC{i}' for i in range(p["n_scs"])]
    p["x0"] = np.zeros((p["T"], 1))  # initial state
    p["input_flows"] = [[[1]] if k<=3 else [[0]] for k in range(p["horizon"])]  # food input for horizon
    return p