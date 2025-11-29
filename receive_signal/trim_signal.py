from src.helpers import plot_signal
from src.configuration import ConfigParams
from src.arinc_constants import PREAMBULE_A, PREAMBULE_M
from src.psk_modem import PSKModem

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from numpy.fft import fft, ifft, fftshift, ifftshift

# configuration
cfg = ConfigParams(fs_hz=16000, fsymb_hz=1800)

# signal receiving
in_rx_signal = np.load("receive_signal/src/arinc_2021_09_15.npy").flatten()
time_s = np.arange(in_rx_signal.shape[0]) / cfg.fs_hz
# plot_signal(in_rx_signal, name="In signal", xlabel="Time [sec]", x_scale_coef=1 / cfg.fs_hz)

# downconversion (1440 Hz + 3 Hz doppler)
signal_base = in_rx_signal * np.exp(-1j * 2 * np.pi * (1440 + 3) * time_s)
x = np.arange(-signal_base.shape[0] // 2, signal_base.shape[0] // 2, 1)
spec_base_signal = fft(signal_base, norm="ortho")
# plt.figure()
# plt.suptitle("spec")
# # plt.plot(x * cfg.fs_hz / signal_base.shape[0], fftshift(np.abs(spec_base_signal)), label="abs")
# plt.plot(np.arange(spec_base_signal.shape[0]), fftshift(np.abs(spec_base_signal)), label="abs")
# plt.xlabel("Freq [hz]")
# plt.ylabel("Spec")
# plt.legend()
# plt.grid()
# plt.show()

new_base_signal_spec = spec_base_signal
up = 1770000
down = 323000
new_base_signal_spec[down:up] = 0

# plt.figure()
# plt.suptitle("spec")
# x = np.arange(-new_base_signal_spec.shape[0] // 2, new_base_signal_spec.shape[0] // 2, 1)
# plt.plot(x * cfg.fs_hz / new_base_signal_spec.shape[0], fftshift(np.abs(new_base_signal_spec)), label="abs")
# plt.xlabel("Freq [hz]")
# plt.ylabel("Spec")
# plt.legend()
# plt.grid()
# plt.show()

# 9/20 for resampling
new_base_signal = ifft(new_base_signal_spec, norm="ortho")
resampled_bb_signal = resample_poly(new_base_signal, 9, 20)

# plt.figure()
# plt.suptitle("spec")
# plt.plot(np.arange(resampled_bb_signal.shape[0]), fftshift(np.abs(fft(resampled_bb_signal, norm="ortho"))), label="abs")
# plt.xlabel("Freq [hz]")
# plt.ylabel("Spec")
# plt.legend()
# plt.grid()
# plt.show()

# preambule detection
fs_hz_new = 7200
num_samples_per_symb = int(fs_hz_new / cfg.fsymb_hz)
print(num_samples_per_symb)
pmod = PSKModem(1)
preambule_modulated = np.repeat(pmod.modulate(PREAMBULE_A), num_samples_per_symb).astype(np.complex128)
# plot_signal(preambule_modulated, name="A")
corr_a = np.correlate(resampled_bb_signal, preambule_modulated, "full")

# m-preambule
preambule_m_modulated = np.repeat(pmod.modulate(PREAMBULE_M), num_samples_per_symb).astype(np.complex128)
# plot_signal(preambule_m_modulated, name="M")
corr_m = np.correlate(resampled_bb_signal, preambule_m_modulated, "full")

plt.figure()
plt.suptitle("Correlation")
plt.plot(np.abs(corr_a), label="A")
plt.plot(np.abs(corr_m), label="M")
plt.xlabel("Samples")
plt.ylabel("Corr")
plt.legend()
plt.grid()
plt.show()

start_sample = 494500
end_sample = 510500

# np.save(
#     "receive_signal/src/trim_s" + str(start_sample) + "_e" + str(end_sample) + "_fs" + str(fs_hz_new) + ".npy",
#     resampled_bb_signal[start_sample:end_sample],
# )
