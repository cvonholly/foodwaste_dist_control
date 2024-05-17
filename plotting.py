import matplotlib.pyplot as plt


def plots(output):
    """
    plots single output and flows
    """
    self_con, foodwaste, flow = [], [], []
    for y in output:
        self_con.append(y[0])
        foodwaste.append(y[1])
        flow.append(y[-1])
    plt.plot(self_con, label=f'self cons., sum: {sum(self_con)}')
    plt.plot(foodwaste, label=f'food waste, sum: {sum(foodwaste)}')
    plt.plot(flow, label=f"input flow, sum: {sum(flow)}")
    plt.legend() 
    plt.show()