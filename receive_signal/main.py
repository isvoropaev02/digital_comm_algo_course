from src.helpers import plot_signal
from src.configuration import ConfigParams
from src.preambule_detector import PreambuleDetector

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

# configuration
cfg = ConfigParams(fs_hz=7200, fsymb_hz=1800)

# signal receiving
in_rx_signal = np.load("receive_signal/src/trim_s69000_e85500_fs7200.npy").flatten()
time_s = np.arange(in_rx_signal.shape[0]) / cfg.fs_hz

# clock synchronization / data rate and interleaver derivation
preamb_detector = PreambuleDetector(fs_hz=cfg.fs_hz, fsymb_hz=cfg.fsymb_hz)
pr_dec_out_signal, data_rate, interleaver_dur = preamb_detector.process(in_rx_signal)

# equalization
