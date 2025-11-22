from src.helpers import plot_signal
from src.configuration import ConfigParams
from src.preambule_detector import PreambuleDetector
from src.equalizer import Equalizer

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

# configuration
cfg = ConfigParams(fs_hz=7200, fsymb_hz=1800)
preamb_detector = PreambuleDetector(fs_hz=cfg.fs_hz, fsymb_hz=cfg.fsymb_hz)
equalizer = Equalizer(mu=0.1, filter_order=15)

# signal receiving
in_rx_signal = np.load("receive_signal/src/trim_s69000_e85500_fs7200.npy").flatten()
time_s = np.arange(in_rx_signal.shape[0]) / cfg.fs_hz

# clock synchronization / data rate and interleaver derivation
pr_dec_out_signal, data_rate, interleaver_dur, shift = preamb_detector.process(in_rx_signal)
plot_signal(pr_dec_out_signal)

# equalization
eq_out_samples = equalizer.process(
    pr_dec_out_signal, shift, int(interleaver_dur * cfg.fsymb_hz / preamb_detector.samples_per_symb)
)
