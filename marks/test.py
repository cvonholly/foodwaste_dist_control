import numpy as np

from flow_factors import flow_matrix_P_to_C, flow_matrix_P_to_SC, flow_matrix_SC_to_C, flow_matrix_C_consumption

T = 5 # number of states 
food_waste = 0.08 # food waste percentage/ratio at producer level
T_start = 3 # when producers start to give food products to charity
T_end = 4 # when producers end to give food products to charity
alpha_start = 0.25 # amount/percentage of food given to charity at beginning
alpha_end = 0.5 # amount/percentage of food given to charity at end
alpha_first_day = 0.2 # probability that food gets consumed from consumer on first day
alpha_last_day = 0.8 # probability that food gets consumed from consumer on last day
f0 = 1 #inflow to producer

# check condition that flow factors of P to C and P to SC fulfill mass condition (e.g. <1)!!!

matrix_P_to_C = flow_matrix_P_to_C(T,food_waste)
print("matrix_P_to_C")
print(matrix_P_to_C)

matrix_P_to_SC = flow_matrix_P_to_SC(T, T_start, T_end, alpha_start, alpha_end)
print("matrix_P_to_SC")
print(matrix_P_to_SC)

matrix_SC_to_C = flow_matrix_SC_to_C(T, T_start, T_end, alpha_start, alpha_end)
print("matrix_SC_to_C")
print(matrix_SC_to_C)

matrix_C_consumption = flow_matrix_C_consumption(T, alpha_first_day, alpha_last_day)
print("matrix_C_consumption")
print(matrix_C_consumption)
