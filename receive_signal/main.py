from src.helpers import plot_signal
from src.configuration import ConfigParams

import numpy as np
import matplotlib.pyplot as plt

# configuration
cfg = ConfigParams(fs_hz=16000, fc_hz=0, f_symb_hz=1800)

# signal receiving
in_rx_signal = np.load("receive_signal/src/arinc_2021_09_15.npy")
print(in_rx_signal.shape)
plot_signal(in_rx_signal, name="In signal", xlabel="Time [sec]", x_scale_coef=1 / cfg.fs_hz)

# preambule detection
# equalization
# demodulation
# deinterleaving
# FEC decoding
