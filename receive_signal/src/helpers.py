import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import *


def plot_signal(signal: np.ndarray, name: str = "", xlabel: str = "x", ylabel: str = "y", x_scale_coef: float = 1) -> None:
    plt.figure(figsize=(7, 9))

    plt.subplot(1, 2, 1)
    plt.suptitle(name)
    x = np.arange(signal.shape[0])
    plt.plot(x * x_scale_coef, np.real(signal), label="Re")
    plt.plot(x * x_scale_coef, np.imag(signal), label="Im")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.suptitle(name + "spectrum")
    f = np.linspace(-1 / 2, 1 / 2, signal.shape[0])
    plt.plot(f / x_scale_coef, np.abs(fftshift(fft(signal, norm="ortho"))))
    plt.xlabel("f")
    plt.ylabel("Abs")
    plt.grid()

    plt.show()
