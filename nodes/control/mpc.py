import numpy as np
import cvxpy as cp


"""
mpc_P moved to producers.py due to import dependencies
"""

class MPC_C:
    def __init__(self, B: np.ndarray,  
        x0: np.ndarray, 
        food_intake: float,
        gammas: np.ndarray,
        u: np.ndarray,
        K: int,
        printme=False):
        """
        creates a MPC object for consumers
        """
        self.B = B
        self.x0 = x0
        self.food_intake = food_intake
        self.gammas = gammas
        self.u = u
        self.K = K
        self.printme = printme
        self.A = None
        self.C = None
    
    def get_A(self, x_store: np.ndarray, x: np.ndarray, n: int):
        x_store, x = np.round(x_store, 4), np.round(x, 4)
        alphas = np.divide(x_store, x, out=np.zeros_like(x_store), where=x!=0)
        A = np.eye(n) * alphas
        A = np.roll(A, 1, axis=0)
        self.A = A
    
    def get_C(self, x_sc: np.ndarray, x_fw: np.ndarray, x: np.ndarray):
        x_sc, x_fw, x = np.round(x_sc, 4), np.round(x_fw, 4), np.round(x, 4)
        thetas = np.divide(x_sc, x, out=np.zeros_like(x_sc), where=x!=0)
        gammas = np.divide(x_fw, x, out=np.zeros_like(x_fw), where=x!=0)
        self.C = np.vstack([thetas, gammas])
    
    def run(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.x0.shape[0]  # Number of states
        ones = np.ones((1,n))

        x = cp.Variable((n, self.K+1), nonneg=True)  # states
        x_store = cp.Variable((n, self.K+1), nonneg=True)
        x_sc = cp.Variable((n, self.K+1), nonneg=True)
        x_fw = cp.Variable((n, self.K+1), nonneg=True)

        # Define the cost function & constraints
        cost = 0
        constraints = []

        # define modified A matrix
        Am = np.roll(np.eye(n), 1, axis=0)
        Am[0, -1] = 0

        for k in range(self.K+1):
            for nn in range(n):
                constraints += [x_fw[nn, k] == self.gammas[nn] * x[nn, k]]  # determine food waste

        for k in range(self.K):
            # state update
            constraints += [x[:, k+1] == Am @ x_store[:, k] + np.resize(self.B @ self.u, n)]  
            # food intake condition
            constraints += [self.food_intake == ones @ x_sc[:, k]]
            # mass-flow balance
            constraints += [x[:, k] == x_store[:, k] + x_sc[:, k] + x_fw[:, k]]  
            constraints += [x[:, k] >= x_store[:, k]]  # mass-flow balance
            constraints += [x[:, k] >= x_sc[:, k]]  # mass-flow balance
            constraints += [x[:, k] >= x_fw[:, k]]  # mass-flow balance

        cost += cp.sum(x_fw)  # cost and terminal cost

        constraints += [x[:, 0] == self.x0]  # Initial condition

        # Solve the optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()
        
        if self.printme:
            for k in range(self.K):
                print(f"=== k={k} ===")
                print("x_store: ", np.round(x_store[:, k].value, 2))
                print("x: ", np.round(x[:, k].value, 2))
                print("x_sc: ", np.round(x_sc[:, k].value,2 ))
                print("x_fw: ", np.round(x_fw[:, k].value, 2))


        if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
            self.get_A(x_store[:, 0].value, x[:, 0].value, n)
            self.get_C(x_sc[:, 0].value, x_fw[:, 0].value, x[:, 0].value)
            return self.A, self.C
        else:
            print("status:", problem.status)
            print("optimal value", problem.value)
            raise ValueError("MPC optimization problem did not solve!")


if __name__=="__main__":
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
    mpc_C = MPC_C(B, x0, food_intake, gammas, u, K, printme=True)
    A, C = mpc_C.run()

    # Print the results
    print("A:", A)
    print("C:", C)
        

