if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np


    def height(xs):
        return np.sin(3 * xs) * .45 + .55


    x_min = -1.2
    x_min = -2
    x_max = 0.6
    X = np.linspace(x_min, x_max, 100)
    Y = np.cos(3*X)
    Y2 = np.sin(np.arctan(-3*np.sin(3*X)))
    Y3 = height(X)

    # plt.plot(X, Y, label="their force(?)")
    # plt.plot(X, Y2, label="my force(?)")
    plt.plot(X, Y3, label="height")
    plt.legend()
    plt.show()
