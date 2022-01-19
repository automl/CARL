if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    R = 1000000  # max resource per config
    eta = np.arange(2, 15)

    base = eta
    s_max = np.floor(np.log(R) / np.log(base))
    B = (s_max + 1) * R

    plt.plot(eta, B, marker='o')
    plt.xlabel("$\eta$")
    plt.ylabel("$B$")
    title = "Select $\eta$ for the maximum budget $B$\nyou would like to spend on the optimization."
    plt.title(title)
    plt.show()
