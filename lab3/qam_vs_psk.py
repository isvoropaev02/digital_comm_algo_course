import numpy as np
from commpy.modulation import PSKModem, QAMModem
import matplotlib.pyplot as plt

MOD_ORDER = 16
NUM_BITS = 4 * 2000
SNR1 = 15 # dB
SNR_RANGE = np.arange(0, 21, 1)

psk = PSKModem(MOD_ORDER)
qam = QAMModem(MOD_ORDER)

# psk.plot_constellation()
# qam.plot_constellation()

bit_message = np.random.randint(low=0, high=2, size=NUM_BITS)
psk_tx_out = psk.modulate(bit_message) / np.sqrt(psk.Es)
qam_tx_out = qam.modulate(bit_message) / np.sqrt(qam.Es)

noise_var = 10**(-SNR1/10)
psk_rx_in = psk_tx_out + np.sqrt(noise_var / 2) * (np.random.randn(psk_tx_out.shape[0]) + 1j * np.random.randn(psk_tx_out.shape[0]))
qam_rx_in = qam_tx_out + np.sqrt(noise_var / 2) * (np.random.randn(qam_tx_out.shape[0]) + 1j * np.random.randn(qam_tx_out.shape[0]))

fig1 = plt.figure(figsize=(12,6))
fig1.suptitle(r'Constellation')
ax1 = fig1.add_subplot(121)

ax1.set_title(f"PSK{MOD_ORDER}")
ax1.scatter(np.real(psk_rx_in), np.imag(psk_rx_in), color="C0")
ax1.set_ylabel("Q")
ax1.set_xlabel("I")
ax1.grid()

ax2 = fig1.add_subplot(122)
ax2.set_title(f"QAM{MOD_ORDER}")
ax2.scatter(np.real(qam_rx_in), np.imag(qam_rx_in), color="C1")
ax2.set_ylabel("Q")
ax2.set_xlabel("I")
ax2.grid()
plt.show()

psk_ber_res = np.zeros(SNR_RANGE.shape, dtype=np.float64)
qam_ber_res = np.zeros(SNR_RANGE.shape, dtype=np.float64)
for j_snr in range(SNR_RANGE.shape[0]):
    bit_message = np.random.randint(low=0, high=2, size=NUM_BITS)
    psk_tx_out = psk.modulate(bit_message) / np.sqrt(psk.Es)
    qam_tx_out = qam.modulate(bit_message) / np.sqrt(qam.Es)

    noise_var = 10**(-SNR_RANGE[j_snr]/10)
    psk_rx_in = psk_tx_out + np.sqrt(noise_var / 2) * (np.random.randn(psk_tx_out.shape[0]) + 1j * np.random.randn(psk_tx_out.shape[0]))
    qam_rx_in = qam_tx_out + np.sqrt(noise_var / 2) * (np.random.randn(qam_tx_out.shape[0]) + 1j * np.random.randn(qam_tx_out.shape[0]))

    psk_rx_bits = psk.demodulate(psk_rx_in * np.sqrt(psk.Es), 'hard')
    qam_rx_bits = qam.demodulate(qam_rx_in * np.sqrt(qam.Es), 'hard')

    psk_ber_res[j_snr] = np.sum(np.abs(psk_rx_bits - bit_message)) / NUM_BITS
    qam_ber_res[j_snr] = np.sum(np.abs(qam_rx_bits - bit_message)) / NUM_BITS

fig2 = plt.figure(figsize=(9,5))
fig2.suptitle(r'BER result')
ax1 = fig2.add_subplot(111)
ax1.plot(SNR_RANGE, psk_ber_res, label=f'PSK{MOD_ORDER}')
ax1.plot(SNR_RANGE, qam_ber_res, label=f'QAM{MOD_ORDER}')
ax1.set_ylabel("BER")
ax1.set_yscale('log')
ax1.set_xlabel("SNR [dB]")
ax1.legend()
ax1.grid()
plt.show()