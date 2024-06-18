import numpy as np


"""
Marc's factors scenario
"""

def get_P_C_facs(T, foodwaste):
    alpha = -T/np.log(foodwaste)
    gamma = 1-gamma