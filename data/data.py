import numpy as np


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


def get_flows(n, T, a):
    """
    get flow factors for this producer

    input:
        n: number flows
        T: number time steps
        a: array of length n for flow factos (constant)
    output:
        flow factor matrix of size n x T
    """
    return np.array([np.array([a for i in range(T)]) for i in range(n)])


def get_P_facs(flows, T, a):
    """
    gets factors for producers

    input:
        flows: matrix representing output flows factors for this Producer

    returns:
        alphas (for A matrix)
            
        self consumption factors
        foodwaste factors
    """
    foodwaste = np.linspace(0, a, T)
    if any(foodwaste > 1):
        print("aborting, invalid factors for c class")
        return False, False
    flows_sum = sum(np.array(flows))  # sum of flows factors. have to be smaller 1
    if len(flows_sum)!=len(foodwaste):
        print("aborting, flows size's do not match")
    print(foodwaste)
    alphas = flows_sum + foodwaste
    return alphas, foodwaste



def get_SC_facs(flows, T, a):
    """
    gets factors for social/charicity actors

    input:
        flows: matrix representing output flows factors for this SC

    returns:
        alphas (for A matrix)
            
        self consumption factors
        foodwaste factors
    """
    foodwaste = np.linspace(0, a, T)
    if any(foodwaste > 1):
        print("aborting, invalid factors for c class")
        return False, False
    flows_sum = sum(flows)  # sum of flows factors. have to be smaller 1
    alphas = flows_sum + foodwaste
    return alphas, foodwaste
