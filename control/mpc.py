import numpy as np
import cvxpy as cp



def mpc_P(A: np.ndarray, 
        B: np.ndarray,  
        q: np.ndarray, 
        x0: np.ndarray, 
        food_input: float,
        N: int) -> np.ndarray:
    """
    Model Predictive Control for producer

    :param A: State transition matrix
    :param B: Control input matrix
    :param q: cost vector
    :param x0: Initial state
    :param N: Prediction horizon
    :return: Optimal control input sequence
    """
    n = A.shape[1]  # Number of states
    m = B.shape[1]  # Number of inputs

    # Define variables
    x = cp.Variable((n, N+1))
    u = cp.Variable((m, N))

    # Define the cost function & constraints
    cost = 0
    constraints = []

    for k in range(N):
        cost += q.T @ x[:, k]
        constraints += [x[:, k+1] == A @ x[:, k] + B @ u[:, k]]

    cost += q.T @ x[:, N] # terminal cost

    constraints += [x[:, 0] == x0]  # Initial condition

    constraints += [cp.sum(u, axis=0) == food_input]  # food input condition

    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
        return u[:, 0].value
    else:
        print("status:", problem.status)
        print("optimal value", problem.value)
        raise ValueError("MPC optimization problem did not solve!")



def get_A(thetas, gammas, dim):
    C = cp.vstack([
        thetas,
        gammas
    ])
    alphas = np.sum(C, axis=0)  # sum along columns
    A = np.eye(dim) - np.eye(dim) * alphas
    A = np.roll(A, 1, axis=0)
    return A


def get_C(thetas, gammas):
    return cp.vstack([thetas, gammas])

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
    thetas = np.array([[cp.Variable() for i in range(n)] for k in range(K+1)])
    # thetas = [cp.Variable((n, 1)) for k in range(K+1)] # food conumption # old

    # Define the cost function & constraints
    cost = 0
    constraints = []

    for k in range(K):
        cost += gammas.T @ x[:, k]
        A = get_A(thetas[k], gammas.T, n)
        constraints += [x[:, k+1] == A @ x[:, k] + B @ u[:, k]]
        constraints += [thetas[k].sum() +  gammas[k] <= 1]
        sigma_k = thetas[k] @ x[:, k]
        constraints += [sigma_k == food_intake]   # food intake condition
        constraints += [gammas[k] + thetas[k].sum() <= 1]   # mass flow condition

    cost += gammas.T @ x[:, K] # terminal cost

    constraints += [x[:, 0] == x0]  # Initial condition

    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    # now all variables are solved and we can extract the values
    A = get_A(thetas[0].value, gammas.T, n)
    C = get_C(thetas[0].value, gammas.T)


    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
        return A, C
    else:
        print("status:", problem.status)
        print("optimal value", problem.value)
        raise ValueError("MPC optimization problem did not solve!")

if __name__=="__main__":
    # Example usage
    # A = np.array([[1.0, 1.0], [0, 1.0]])
    # B = np.array([[0], [1.0]])
    # q = np.zeros((2,1))
    # q[-1] = 1
    # x0 = np.array([0, 0])
    # N = 10

    # u_optimal = mpc_P(A, B, q, x0, 2, N)
    # print(f"The optimal control input at the current step is: {u_optimal}")

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
