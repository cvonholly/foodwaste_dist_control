import numpy as np
import pandas as pd
import cvxpy as cp
import sys

from node import Node
import control.mpc


def get_A(thetas, gammas, dim):
    Ag = np.eye(dim) - np.eye(dim) * gammas
    Ag = np.roll(Ag, 1, axis=0)
    At = cp.diag()
    return At, Ag

def get_C(thetas, gammas):
    return cp.vstack([thetas, gammas])

def mpc_C(B: np.ndarray,  
        x0: np.ndarray, 
        food_intake: float,
        gammas: np.ndarray,
        u: np.ndarray,
        K: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Model Predictive Control for consumer

    :param B: Control input matrix
    :param x0: Initial state
    :param food_intake: food intake required for every time step
    :param gammas: food waste factors
    :param u: input flow
    :param K: Prediction horizon

    :return: A, C matrices of system
    """
    n = x0.shape[0]  # Number of states

    # x = cp.Variable((n, K+1))  # states
    x = np.array([[cp.Variable() for i in range(n)] for k in range(K+1)])
    thetas = np.array([[cp.Variable() for i in range(n)] for k in range(K+1)])
    # thetas = [cp.Variable((n, 1)) for k in range(K+1)] # food conumption # old

    # Define the cost function & constraints
    cost = 0
    constraints = []
    """
    reflect: THIS HAS WORJED BEFORE !!
    
    just use a previous version and fix it !
    """

    for k in range(K):
        cost += gammas @ x[k]
        At, Ag = get_A(thetas[k], gammas, n)
        print(At)
        print(Ag)

        constraints += [x[k+1] == At @ x[k] +  Ag @ x[k] + B @ u[:, k]]
        constraints += [thetas[k].sum() +  gammas[k] <= 1]
        sigma_k = thetas[k] @ x[:, k]
        constraints += [sigma_k == food_intake]   # food intake condition
        constraints += [gammas[k] + thetas[k].sum() <= 1]   # mass flow condition

    cost += gammas @ x[:, K] # terminal cost

    constraints += [x[:, 0] == x0]  # Initial condition

    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    # now all variables are solved and we can extract the values
    A = get_A(thetas[0].value, gammas, n)
    C = get_C(thetas[0].value, gammas)


    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
        return A, C
    else:
        print("status:", problem.status)
        print("optimal value", problem.value)
        raise ValueError("MPC optimization problem did not solve!")


class C(Node):
    """
    class representing consumers
    """
    def __init__(self, name: str, T: int,
                 fw_matrix: np.ndarray,
                 n_flows: int, 
                 x0: np.ndarray,
                 ec_mpc=False,
                 food_intake=0.0,
                 mpc_h=5) -> None:
        """
        inputs (see above)
        """
        self.name = name
        self.T = T
        self.n = x0.size
        self.sc_matrix = np.zeros((1, self.n))
        self.fw_matrix = fw_matrix
        self.n_flows = n_flows
        self.x0 = x0   # x0 state
        self.x = x0   # current state
        self.y = None   # output
        self.x_hist = []  # previous x's
        self.y_names = ['self consumption', 'food waste']  # output names
        self.alphas = np.ndarray  # initialize alphas
        self.C = self.get_C()  # get output matrix
        if (self.alphas > 1).any():
            raise Exception("aborting, alpha value is greater 0")        
        self.A = self.get_A()
        self.B = None  # is computed at first time step (due to wait for input)
        self.ec_mpc = ec_mpc  # bool weather to use economic mpc
        self.food_intake = food_intake  # daily food intake required
        self.mpc_h = mpc_h
    
    def get_B(self, inputs: np.array):
        """
        inputs: input flows
            rows: nodes (n)
            cols: time step inputs (T)
        """
        self.n_u = inputs.shape  # = (n, T) = (#nodes, #states)
        self.B = np.hstack([np.eye(self.n_u[1]) for i in range(self.n_u[0])])  # hor. stacked identity matrices
        return self.B
    
    def get_C(self):
        C = np.vstack((self.sc_matrix, self.fw_matrix))
        self.alphas = C.sum(axis=0)  # alpha values for A matrix
        if (self.alphas > 1).any():
            raise Exception("aborting, alpha value is greater 0")
        return C
    
    def get_A_C(self):
        """
        we need to adapt A and C at every time step in order to ensure our
        consumer "eats enough": sum(gamma * x) == food_intake
        """
        current_food = self.x[:-1].sum()  # exclude final stage as this goes to waste
        if current_food < self.food_intake:
            print("ABORTING")
            print("consumer ", self.name, " need food intake ", self.food_intake, " but only has in stock ", current_food)
            print("x: ", self.x)
            raise Exception("aborting, not enough food")
        gamma = (current_food - self.food_intake) / (self.n - 1)
        self.sc_matrix = np.array([gamma for i in range(self.n)])
        self.sc_matrix[-1] = 0   # at final stage, everything goes to waste
        self.C = self.get_C()  # computes self.C, self.alphas
        self.A = self.get_A()

    def sim_step(self, k, inputs: pd.DataFrame):
        inputs = inputs.fillna(0).to_numpy()  # inputs to numpy
        if self.B is None: self.B = self.get_B(inputs)  # computes B at first time step
        inputs = np.resize(inputs, (self.B.shape[1], 1))  # resize for matrix multiplication
        self.x_hist.append(self.x)
        if k>0:
            if self.ec_mpc:  # compute optimal A, C if using MPC
                self.A, self.C = mpc_C(self.B, self.x, self.food_intake, self.fw_matrix, inputs, self.mpc_h)
            else:
                self.get_A_C()  # computes self.C, self.alphas, self.A
        if k==3:
            self.print_all()
        self.x = self.A @ self.x + self.B @ inputs  # time step
        self.y = self.C @ self.x   # get output
        return self.y
        
if __name__=="__main__":
    """
    only for testing
    """
    # prediction horizon
    K = 4

    # Initial state (x0) - Example with 2 states
    x0 = np.array([1, 1, 1])

    # Control input matrix (B) - Example with 2 states and 2 control inputs
    B = np.array([1, 1])
    B = np.vstack((B, np.array([[0, 0] for i in range(x0.size-1)]))) 

    # Food intake required for every time step (food_intake)
    food_intake = 0.1

    # Food waste factors (gammas)
    gammas = np.array([.1, .2, 1])

    # Input flow (u) - Example with 2 control inputs
    u = np.array([[1],
                  [1]])
    

    # create MPC
    mpc_C =  control.mpc.MPC_C(B, x0, food_intake, gammas, u, K, printme=True)
    A, C = mpc_C.run()

    # Print the results
    print("A:", A)
    print("C:", C)
