import numpy as np


def flow_to_P(T,f0):
    flow_vector = np.zeros(T)
    flow_vector[0] = f0
    return flow_vector

def flow_from_P_to_C(flow_matrix_P_to_C, P_state):
    flow = np.dot(flow_matrix_P_to_C, P_state)
    return flow

def flow_from_P_to_SC(flow_matrix_P_to_SC, P_state):
    flow = np.dot(flow_matrix_P_to_SC, P_state)
    return flow

def flow_from_SC_to_C(flow_matrix_SC_to_C, SC_state):
    flow = np.dot(flow_matrix_SC_to_C, SC_state)
    return flow

def flow_C_consumption(flow_matrix_C_consumption, C_state):
    flow = np.dot(flow_matrix_C_consumption, C_state)
    return flow

def food_waste_P(T,P_state):

    food_waste_matrix = np.zeros((T,T))
    food_waste_matrix[(T-1),(T-1)] = 1

    food_waste_P = np.dot(food_waste_matrix, P_state)
    return food_waste_P

def food_waste_SC(T,SC_state):

    food_waste_matrix = np.zeros((T,T))
    food_waste_matrix[(T-1),(T-1)] = 1

    food_waste_SC = np.dot(food_waste_matrix, SC_state)
    return food_waste_SC

def food_waste_C(T,C_state):

    food_waste_matrix = np.zeros((T,T))
    food_waste_matrix[(T-1),(T-1)] = 1

    food_waste_C = np.dot(food_waste_matrix, C_state)
    return food_waste_C


