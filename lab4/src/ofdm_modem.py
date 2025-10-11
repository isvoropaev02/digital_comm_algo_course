import numpy as np
from src.dpsk_modem import DPSKModem

class OFDMModem:
    def __init__(self, num_sc: int = 1) -> None:
        self.__num_sc = num_sc
        # self.__cp_len = int(num_sc * 4.69 / 71.35) # https://www.sharetechnote.com/html/5G/5G_FrameStructure.html
        self.__cp_len = max(int(num_sc * 0.2), 1)
    
    @property
    def num_sc(self) -> int:
        return self.__num_sc
    
    @property
    def cp_len(self) -> int:
        return self.__cp_len
    

    def ofdm_modulate(self, samples: np.ndarray) -> np.ndarray:
        td_signal = np.fft.ifft(samples, norm='ortho')
        return np.concat([td_signal[len(td_signal)-self.__cp_len:], td_signal])
    
    def ofdm_demodulate(self, signal: np.ndarray) -> np.ndarray:
        # trimed_signal = signal[(self.__cp_len // 2) : len(signal) + (-self.__cp_len // 2)]
        trimed_signal = signal[self.__cp_len:]
        return np.fft.fft(trimed_signal, norm='ortho')
    
    def process_tx(self, in_bits: np.ndarray, modem_block: DPSKModem, num_symb: int) -> np.ndarray:
        bits_per_symb = modem_block.bits_per_sample * (self.__num_sc - 1)
        if len(in_bits) != bits_per_symb*num_symb:
            raise ValueError(f"Invalid len(in_bits)={len(in_bits)}, the value should be equal to {bits_per_symb*num_symb}")
        tx_out_signal = np.array([], dtype=np.complex128)
        for j_symb in range(num_symb):
            fd_samples = modem_block.modulate(in_bits[j_symb * bits_per_symb : (j_symb+1) * bits_per_symb])
            tx_out_signal = np.concatenate([tx_out_signal, self.ofdm_modulate(fd_samples)])
        return tx_out_signal
    
    def process_rx(self, rx_signal: np.ndarray, modem_block: DPSKModem, num_symb: int) -> np.ndarray:
        samples_per_symb = self.__num_sc + self.__cp_len
        rx_out_bits = np.array([], dtype=np.int8)
        for j_symb in range(num_symb):
            fd_samples = self.ofdm_demodulate(rx_signal[j_symb * samples_per_symb : (j_symb+1) * samples_per_symb])
            rx_out_bits = np.concatenate([rx_out_bits, modem_block.demodulate(fd_samples)])
        return rx_out_bits
    


if __name__ == "__main__":
    num_sc = 4
    ofdm_block = OFDMModem(num_sc=num_sc)
    print("num_sc: ", ofdm_block.num_sc)
    print("cp_len: ", ofdm_block.cp_len)
    test_signal = ((1 - 2*np.random.randint(low=0, high=2, size=num_sc))
                   + 1j * (1 - 2*np.random.randint(low=0, high=2, size=num_sc))) / np.sqrt(2)
    
    print("test samples: ", (test_signal))
    
    td_signal = ofdm_block.ofdm_modulate(test_signal)
    print("td signal: ", (td_signal))

    samples_received = ofdm_block.ofdm_demodulate(td_signal)
    print("rec samples: ", (samples_received))
    print(f"Correct: {np.allclose(samples_received, test_signal, atol=1e-10)}")