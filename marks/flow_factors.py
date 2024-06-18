import numpy as np

        
def flow_matrix_P_to_C(T,food_waste):
        
        alpha = -T/(np.log(food_waste))
        matrix_factors = 1 - (np.exp(-1/alpha))
        flow_matrix = np.zeros((T,T))

        for i in range(T-1):
            flow_matrix[i,i] = matrix_factors

        return flow_matrix
        

def flow_matrix_P_to_SC(T, T_start, T_end, alpha_start, alpha_end):
        
        alpha = (alpha_end-alpha_start)/(T_end-T_start)
        flow_matrix = np.zeros((T,T))
        start_index = T_start-1
        stop_index = T_end-1
        step = 1
        for i in range((T_end-T_start)+1):
            flow_matrix[(i+start_index),(i+start_index)] = alpha_start + (i*alpha)

        return flow_matrix


def flow_matrix_SC_to_C(T, T_start, T_end, alpha_start, alpha_end):
        
        alpha = (alpha_end-alpha_start)/(T_end-T_start)
        flow_matrix = np.zeros((T,T))
        start_index = T_start-1
        stop_index = T_end-1
        step = 1
        for i in range((T_end-T_start)+1):
            flow_matrix[(i+start_index),(i+start_index)] = alpha_start + (i*alpha)

        return flow_matrix


def flow_matrix_C_consumption(T, alpha_first_day, alpha_last_day):
        
        alpha = (T-1)/(np.log(alpha_last_day-alpha_first_day+1))
        flow_matrix = np.zeros((1,T))
      
        for i in range(T):
            flow_matrix[0, i] = alpha_first_day + (np.exp(i/alpha)-1)
        
        flow_matrix[0, -1] = 0  # at final time step this has to be zero

        return flow_matrix





    