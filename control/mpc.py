import numpy as np
import cvxpy as cp


"""
mpc_P moved to producers.py due to import dependencies
"""


def get_A(x_store: np.ndarray, x: np.ndarray, n: int):
    # need to round in order to preserve results
    x_store, x = np.round(x_store, 4), np.round(x, 4)
    print("x_store: ", x_store)
    print("x: ", x)
    alphas = np.divide(x_store, x, out=np.zeros_like(x_store), where=x!=0)
    print("alphas: ", alphas)
    A = np.eye(n) - np.eye(n) * alphas
    A = np.roll(A, 1, axis=0)
    return A

def get_C(x_sc: np.ndarray, x_fw: np.ndarray, x: np.ndarray):
    x_sc, x_fw, x = np.round(x_sc, 4), np.round(x_fw, 4), np.round(x, 4)
    
    print("x_sc: ", x_sc)
    print("x_fw: ", x_fw)
    thetas = np.divide(x_sc, x, out=np.zeros_like(x_sc), where=x!=0)
    gammas = np.divide(x_fw, x, out=np.zeros_like(x_fw), where=x!=0)

    # thetas = 1 - (x-x_sc) / x
    # gammas = 1 - (x-x_fw) / x
    return np.vstack([thetas, gammas])

def mpc_C(B: np.ndarray,  
        x0: np.ndarray, 
        food_intake: float,
        gammas: np.ndarray,
        u: np.ndarray,
        K: int,
        printme=False) -> tuple[np.ndarray, np.ndarray]:
    """
    Model Predictive Control for producer

    :param B: Control input matrix (n x m)
    :param x0: Initial state
    :param food_intake: food intake required for every time step
    :param gammas: food waste factors
    :param u: input flow (m)
    :param K: Prediction horizon

    :return: A, C
    """
    n = x0.shape[0]  # Number of states
    ones = np.ones((1,n))

    x = cp.Variable((n, K+1), nonneg=True)  # states
    x_store = cp.Variable((n, K+1), nonneg=True)
    x_sc = cp.Variable((n, K+1), nonneg=True)
    x_fw = cp.Variable((n, K+1), nonneg=True)

    # Define the cost function & constraints
    cost = 0
    constraints = []

    # define modified A matrix
    Am = np.roll(np.eye(n), 1, axis=0)
    Am[0, -1] = 0

    for k in range(K+1):
        constraints += [x_fw[:, k] == gammas.T @ x[:, k]]  # determine food waste

    for k in range(K):
        # state update
        constraints += [x[:, k+1] == Am @ x_store[:, k] + np.resize(B @ u, n)]  
        # food intake condition
        constraints += [food_intake == ones @ x_sc[:, k]]
        # mass-flow balance
        constraints += [x[:, k] == x_store[:, k] + x_sc[:, k] + x_fw[:, k]]  
        constraints += [x[:, k] >= x_store[:, k]]  # mass-flow balance
        constraints += [x[:, k] >= x_sc[:, k]]  # mass-flow balance
        constraints += [x[:, k] >= x_fw[:, k]]  # mass-flow balance

    cost += cp.sum(x_fw)  # cost and terminal cost

    constraints += [x[:, 0] == x0]  # Initial condition

    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()
    
    if printme:
        for k in range(K):
            print(f"=== k={k} ===")
            print("x_store: ", x_store[:, k].value)
            print("x: ", x[:, k].value)
            print("x_sc: ", x_sc[:, k].value)
            print("x_fw: ", x_fw[:, k].value)


    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
        A, C = get_A(x_store[:, 0].value, x[:, 0].value, n), get_C(x_sc[:, 0].value, x_fw[:, 0].value, x[:, 0].value)
        return A, C
    else:
        print("status:", problem.status)
        print("optimal value", problem.value)
        raise ValueError("MPC optimization problem did not solve!")

if __name__=="__main__":
    # prediction horizon
    K = 4

    # Initial state (x0) - Example with 2 states
    x0 = np.array([1, 0, 0])

    # Control input matrix (B) - Example with 2 states and 2 control inputs
    B = np.array([1, 1])
    B = np.vstack((B, np.array([[0, 0] for i in range(x0.size-1)]))) 

    # Food intake required for every time step (food_intake)
    food_intake = 0.1

    # Food waste factors (gammas)
    gammas = np.array([0, 0, 1])

    # Input flow (u) - Example with 2 control inputs
    u = np.array([[1],
                  [1]])

    

    # Invoke the function
    A, C = mpc_C(B, x0, food_intake, gammas, u, K, printme=True)

    # Print the results
    print("A:", A)
    print("C:", C)
