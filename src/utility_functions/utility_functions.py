import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_fitness():
    data = pd.read_csv('a_file.txt', sep=',', header=None)
    data = pd.DataFrame(data)
    plt.xlabel("Generations")
    plt.ylabel("Evaluated RMSE")
    # plt.plot(c)
    plt.title('RMSE')
    x = data[0]
    y = data[1]

    ax = plt.figure().gca()
    ax.set_ylabel('Best Genome RMSE')
    ax.set_title('RMSE over Generations')
    ax.set_xlabel('Generations')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(x, y)
    plt.show()
    return

if __name__ == '__main__':
    plot_fitness()

