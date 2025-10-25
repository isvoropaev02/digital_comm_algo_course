import numpy as np
import matplotlib.pyplot as plt


def plot_signal(signal: np.ndarray, name: str = "", xlabel: str = "x", ylabel: str = "y", x_scale_coef: float = 1) -> None:
    plt.figure()
    plt.suptitle(name)
    x = np.arange(signal.shape[0])
    plt.plot(x * x_scale_coef, np.real(signal), label="Re")
    plt.plot(x * x_scale_coef, np.imag(signal), label="Im")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()
