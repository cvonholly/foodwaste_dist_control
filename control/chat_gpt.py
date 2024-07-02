import numpy as np
import cvxpy as cp

"""
mpc_P moved to producers.py due to import dependencies
"""

def get_A(thetas, gammas, dim):
    C = np.vstack([
        thetas,
        gammas
    ])
    alphas = np.sum(C, axis=0)  # sum along columns
    A = np.eye(dim) - np.eye(dim) * alphas
    A = np.roll(A, 1, axis=0)
    return A

def get_C(thetas, gammas):
    return np.vstack([thetas, gammas])

def mpc_C(B: np.ndarray,  
        x0: np.ndarray, 
        food_intake: float,
        gammas: np.ndarray,
        u: np.ndarray,
        K: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Model Predictive Control for producer

    :param B: Control input matrix
    :param x0: Initial state
    :param food_intake: food intake required for every time step
    :param gammas: food waste factors
    :param u: input flow
    :param K: Prediction horizon

    :return: A, C
    """
    n = x0.shape[0]  # Number of states

    x = cp.Variable((n, K+1))  # states
    thetas = [cp.Variable(n) for k in range(K+1)]  # food consumption

    # Define the cost function & constraints
    cost = 0
    constraints = []

    # for identity matrix
    ones = cp.Variable(n)
    cp_eye = cp.diag(ones)
    constraints += [ones == np.ones(n)]

    for k in range(K):
        cost += gammas.T @ x[:, k]
        # Construct constraints with CVXPY variables
        constraints += [x[:, k+1] == (cp_eye - cp.diag(thetas[k] + gammas)) @ x[:, k] + B @ u[:, k]]
        constraints += [cp.sum(thetas[k]) + gammas[k] <= 1]
        sigma_k = cp.sum(thetas[k] * x[:, k])
        constraints += [sigma_k == food_intake]  # food intake condition
        constraints += [gammas[k] + cp.sum(thetas[k]) <= 1]  # mass flow condition

    cost += gammas.T @ x[:, K]  # terminal cost

    constraints += [x[:, 0] == x0]  # Initial condition

    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
        # now all variables are solved and we can extract the values
        thetas_values = np.array([thetas[k].value for k in range(K+1)])
        A = get_A(thetas_values[0], gammas, n)
        C = get_C(thetas_values[0], gammas)
        return A, C
    else:
        print("status:", problem.status)
        print("optimal value", problem.value)
        raise ValueError("MPC optimization problem did not solve!")

if __name__=="__main__":
    # Example usage
    # Control input matrix (B) - Example with 2 states and 2 control inputs
    B = np.array([[0.1, 0.2],
                  [0.3, 0.4]])

    # Initial state (x0) - Example with 2 states
    x0 = np.array([1.0, 0.5])

    # Food intake required for every time step (food_intake)
    food_intake = 0.3

    # Food waste factors (gammas) - Example with a prediction horizon of 3
    gammas = np.array([0.1, 0.2])

    # Input flow (u) - Example with 2 control inputs and a prediction horizon of 3
    u = np.array([[0.05, 0.06, 0.07],
                  [0.08, 0.09, 0.1]])

    # Prediction horizon (K)
    K = 3

    # Invoke the function
    A, C = mpc_C(B, x0, food_intake, gammas, u, K)

    # Print the results
    print("A:", A)
    print("C:", C)
