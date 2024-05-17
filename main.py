from data import *


if __name__=="__main__":
    T = 5
    alpha = .5
    C1 = C(T, alpha, 2, np.zeros((1,T)).T)
    C1.print()

    S = Simulation([C1], [], [])
    S.sim_step()