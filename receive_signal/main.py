from src.helpers import plot_signal
from src.configuration import ConfigParams
from src.preambule_detector import PreambuleDetector
from src.equalizer import Equalizer
from src.decoder_chain_rate300 import Rate300DecChain

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

# configuration
cfg = ConfigParams(fs_hz=7200, fsymb_hz=1800)
preamb_detector = PreambuleDetector(fs_hz=cfg.fs_hz, fsymb_hz=cfg.fsymb_hz)
equalizer = Equalizer(mu=0.01, filter_order=15)
dec_chain = Rate300DecChain()

# signal receiving
in_rx_signal = np.load("receive_signal/src/trim_s69000_e85500_fs7200.npy").flatten()
time_s = np.arange(in_rx_signal.shape[0]) / cfg.fs_hz

# clock synchronization / data rate and interleaver derivation
pr_dec_out_signal, data_rate, interleaver_dur, shift = preamb_detector.process(in_rx_signal)
# plot_signal(pr_dec_out_signal)
print(pr_dec_out_signal.shape)
print(int(interleaver_dur / 25 * 45))

# equalization
# eq_out_samples = equalizer.process(pr_dec_out_signal, shift, int(interleaver_dur / 25 * 45))
# plt.figure()
# plt.scatter(np.real(eq_out_samples), np.imag(eq_out_samples))
# plt.show()
equalizer.run_ut()
