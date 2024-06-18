import numpy as np


def get_simple_fw_matrix(flow: np.ndarray):
    """
    flow: output flow
    returns food waste at final time step
    """
    fw_row = np.zeros((1, flow.shape[1]))
    fw_row[0, -1] = 1  # pat final time, everything goes to waste
    return fw_row